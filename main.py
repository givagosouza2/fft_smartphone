import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import ConvexHull
from scipy.stats import entropy as scipy_entropy, gaussian_kde
from scipy.ndimage import maximum_filter

st.set_page_config(page_title='Finger Tapping Test — Análise Quantitativa', layout='wide')

EPS = 1e-12

# -------------------------
# Helpers
# -------------------------
def read_ftt_file(uploaded_file):
    raw = uploaded_file.getvalue()
    last_err = None
    # Try common encodings and delimiters.
    for enc in ('utf-8-sig', 'utf-8', 'latin1'):
        try:
            text = raw.decode(enc)
        except Exception as e:
            last_err = e
            continue
        for sep in (',', ';', '\t'):
            try:
                tmp = pd.read_csv(io.StringIO(text), sep=sep, engine='python')
                if tmp.shape[1] >= 4:
                    break
            except Exception as e:
                last_err = e
                tmp = None
        if tmp is not None and tmp.shape[1] >= 4:
            break
    else:
        raise ValueError(f'Não foi possível ler o arquivo: {last_err}')

    # Always use the first four columns, regardless of their headers.
    df = tmp.iloc[:, :4].copy()
    df.columns = ['TEMPO', 'X', 'Y', 'AREA']
    for c in ['TEMPO', 'X', 'Y', 'AREA']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['TEMPO', 'X', 'Y']).reset_index(drop=True)
    df = df.sort_values('TEMPO').drop_duplicates(subset=['TEMPO'], keep='first').reset_index(drop=True)
    return df


def shannon_entropy(values, bins='fd'):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2 or np.allclose(values, values[0]):
        return 0.0
    counts, _ = np.histogram(values, bins=bins)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(scipy_entropy(p, base=2))


def normalized_shannon(values, bins='fd'):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2 or np.allclose(values, values[0]):
        return 0.0
    counts, _ = np.histogram(values, bins=bins)
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    p = counts / counts.sum()
    h = scipy_entropy(p, base=2)
    return float(h / np.log2(len(counts)))


def sample_entropy(x, m=2, r=None):
    """Simple SampEn implementation suitable for short FTT series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < m + 3:
        return np.nan
    if r is None:
        sd = np.std(x, ddof=1)
        if sd <= EPS:
            return 0.0
        r = 0.2 * sd

    def _count(mm):
        templates = np.array([x[i:i+mm] for i in range(n-mm+1)])
        count = 0
        total = 0
        for i in range(len(templates)-1):
            d = np.max(np.abs(templates[i+1:] - templates[i]), axis=1)
            count += np.sum(d <= r)
            total += len(d)
        return count, total

    b, tb = _count(m)
    a, ta = _count(m+1)
    if b == 0 or a == 0:
        return np.nan
    return float(-np.log((a / ta) / (b / tb)))


def spatial_grid_entropy(x, y, grid_n=8):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan, np.nan
    # Fixed number of cells makes values easier to compare across trials.
    xr = np.ptp(x); yr = np.ptp(y)
    if xr <= EPS and yr <= EPS:
        return 0.0, 0.0
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if xr <= EPS:
        xmin -= .5; xmax += .5
    if yr <= EPS:
        ymin -= .5; ymax += .5
    h2, _, _ = np.histogram2d(x, y, bins=grid_n, range=[[xmin, xmax], [ymin, ymax]])
    counts = h2.ravel()
    counts = counts[counts > 0]
    p = counts / counts.sum()
    h = float(scipy_entropy(p, base=2))
    hnorm = float(h / np.log2(grid_n * grid_n)) if grid_n > 1 else 0.0
    return h, hnorm


def vector_anisotropy(vectors, normalize=False):
    """Second-moment ellipse of vectors joined at the origin.

    Raw vectors -> magnitude-weighted anisotropy.
    Unit vectors -> direction-only anisotropy.
    Returns axis ratio, major/minor semi-axis proxies, angle, eigenvectors.
    """
    V = np.asarray(vectors, dtype=float)
    if V.ndim != 2 or V.shape[0] < 2:
        return dict(ratio=np.nan, major=np.nan, minor=np.nan, angle=np.nan, eigvals=None, eigvecs=None)
    norms = np.linalg.norm(V, axis=1)
    V = V[norms > EPS]
    if len(V) < 2:
        return dict(ratio=np.nan, major=np.nan, minor=np.nan, angle=np.nan, eigvals=None, eigvecs=None)
    if normalize:
        V = V / np.linalg.norm(V, axis=1, keepdims=True)

    # Common origin is preserved: use second moment around zero, not covariance around vector mean.
    M = (V.T @ V) / len(V)
    eigvals, eigvecs = np.linalg.eigh(M)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    eigvals = np.maximum(eigvals, 0)
    major = np.sqrt(eigvals[0])
    minor = np.sqrt(eigvals[1])
    ratio = major / minor if minor > EPS else np.inf
    vec = eigvecs[:, 0]
    angle = float(np.degrees(np.arctan2(vec[1], vec[0])) % 180)
    return dict(ratio=float(ratio), major=float(major), minor=float(minor), angle=angle,
                eigvals=eigvals, eigvecs=eigvecs)


def point_ellipse_metrics(x, y):
    P = np.column_stack([x, y]).astype(float)
    if len(P) < 3:
        return dict(major=np.nan, minor=np.nan, ratio=np.nan, eccentricity=np.nan,
                    angle=np.nan, area95=np.nan)
    C = np.cov(P, rowvar=False, ddof=1)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.maximum(vals, 0)
    # 95% bivariate normal contour: chi-square df=2, p=.95 = 5.991
    k = math.sqrt(5.991)
    major = k * math.sqrt(vals[0])
    minor = k * math.sqrt(vals[1])
    ratio = major / minor if minor > EPS else np.inf
    ecc = math.sqrt(max(0.0, 1 - (minor**2 / major**2))) if major > EPS else 0.0
    v = vecs[:, 0]
    angle = float(np.degrees(np.arctan2(v[1], v[0])) % 180)
    area = float(math.pi * major * minor)
    return dict(major=float(major), minor=float(minor), ratio=float(ratio),
                eccentricity=float(ecc), angle=angle, area95=area)


def circular_metrics(angles_rad):
    a = np.asarray(angles_rad, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    z = np.mean(np.exp(1j * a))
    r = abs(z)
    mean_angle = np.degrees(np.angle(z)) % 360
    circ_var = 1 - r
    return float(mean_angle), float(r), float(circ_var)


def angular_entropy(angles_rad, bins=18):
    a = np.asarray(angles_rad, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan
    a = a % (2*np.pi)
    counts, _ = np.histogram(a, bins=np.linspace(0, 2*np.pi, bins+1))
    counts = counts[counts > 0]
    p = counts / counts.sum()
    h = float(scipy_entropy(p, base=2))
    hn = float(h / np.log2(bins))
    return h, hn


def linear_slope(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.ptp(x[mask]) <= EPS:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def lag1_autocorr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.std(x[:-1]) <= EPS or np.std(x[1:]) <= EPS:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0,1])


def safe_cv(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or abs(np.mean(x)) <= EPS:
        return np.nan
    return float(100*np.std(x, ddof=1)/np.mean(x))



def kde2d_metrics(x, y, grid_size=180, bw_method='scott', levels=(0.50,0.75,0.90,0.95),
                  target_x=None, target_y=None, target_radius=None):
    """2D Gaussian KDE plus quantitative HDR metrics.

    HDR areas are the smallest grid areas containing the requested probability mass.
    Values are returned together with the evaluated grid for plotting.
    """
    x=np.asarray(x,float); y=np.asarray(y,float)
    mask=np.isfinite(x)&np.isfinite(y)
    x=x[mask]; y=y[mask]
    if len(x)<3 or np.ptp(x)<=EPS or np.ptp(y)<=EPS:
        return None

    # modest padding avoids clipping density tails at the observed extrema
    padx=max(np.ptp(x)*0.15, 1.0); pady=max(np.ptp(y)*0.15, 1.0)
    xmin,xmax=x.min()-padx,x.max()+padx
    ymin,ymax=y.min()-pady,y.max()+pady
    gx=np.linspace(xmin,xmax,grid_size); gy=np.linspace(ymin,ymax,grid_size)
    Xg,Yg=np.meshgrid(gx,gy)
    pos=np.vstack([Xg.ravel(),Yg.ravel()])
    try:
        kde=gaussian_kde(np.vstack([x,y]), bw_method=bw_method)
        Z=kde(pos).reshape(grid_size,grid_size)
    except Exception:
        return None

    dxg=gx[1]-gx[0]; dyg=gy[1]-gy[0]; cell_area=dxg*dyg
    mass=Z*cell_area
    total_mass=mass.sum()
    if total_mass<=EPS:
        return None
    mass=mass/total_mass
    Znorm=Z/total_mass

    imax=np.unravel_index(np.argmax(Znorm),Znorm.shape)
    mode_x=float(Xg[imax]); mode_y=float(Yg[imax]); peak=float(Znorm[imax])
    mean_density=float(np.mean(Znorm))
    peak_mean=float(peak/mean_density) if mean_density>EPS else np.nan

    flatZ=Znorm.ravel(); flatM=mass.ravel()
    order=np.argsort(flatZ)[::-1]
    cum=np.cumsum(flatM[order])
    hdr_areas={}; thresholds={}
    for lev in levels:
        k=int(np.searchsorted(cum,lev,side='left'))
        k=min(k,len(order)-1)
        thr=float(flatZ[order[k]])
        thresholds[lev]=thr
        hdr_areas[lev]=float(np.sum(Znorm>=thr)*cell_area)

    # Differential entropy of the discretized KDE density (bits).
    p=mass.ravel(); p=p[p>0]
    kent=float(-np.sum(p*np.log2(p)))

    # Count prominent local maxima; threshold at 5% of global peak and separate by ~5 grid cells.
    neighborhood=11
    local=(Znorm==maximum_filter(Znorm,size=neighborhood,mode='nearest'))
    prominent=local & (Znorm>=0.05*peak)
    n_peaks=int(np.sum(prominent))

    out={
        'KDE - densidade máxima (1/px²)':peak,
        'KDE - modo X (px)':mode_x,
        'KDE - modo Y (px)':mode_y,
        'KDE - razão pico/média':peak_mean,
        'KDE - entropia espacial discreta (bits)':kent,
        'KDE - número de picos proeminentes':n_peaks,
    }
    for lev in levels:
        out[f'KDE - área HDR {int(lev*100)}% (px²)']=hdr_areas[lev]

    if target_x is not None and target_y is not None:
        out['KDE - distância do modo ao centro do alvo (px)']=float(np.hypot(mode_x-target_x,mode_y-target_y))
        # density at target centre
        try:
            out['KDE - densidade no centro do alvo (1/px²)']=float(kde(np.array([[target_x],[target_y]]))[0]/total_mass)
        except Exception:
            out['KDE - densidade no centro do alvo (1/px²)']=np.nan
        if target_radius is not None and target_radius>0:
            inside=((Xg-target_x)**2+(Yg-target_y)**2)<=target_radius**2
            out['KDE - massa de probabilidade dentro do alvo (%)']=float(100*mass[inside].sum())

    return {'metrics':out,'gx':gx,'gy':gy,'X':Xg,'Y':Yg,'Z':Znorm,'mass':mass,
            'thresholds':thresholds,'mode':(mode_x,mode_y),'peak_mask':prominent}


def kde_overlap(kde_a, kde_b):
    """Histogram-intersection overlap of two KDEs evaluated on the same grid."""
    if kde_a is None or kde_b is None:
        return np.nan
    if kde_a['mass'].shape != kde_b['mass'].shape:
        return np.nan
    return float(np.minimum(kde_a['mass'],kde_b['mass']).sum())


def temporal_kde_comparison(df, grid_size=160, bw_method='scott'):
    """Early/middle/late KDEs on one common grid, returning drift/area/overlap metrics."""
    n=len(df)
    if n<9:
        return None
    parts=np.array_split(np.arange(n),3)
    x=df['X'].to_numpy(float); y=df['Y'].to_numpy(float)
    padx=max(np.ptp(x)*0.15,1.0); pady=max(np.ptp(y)*0.15,1.0)
    gx=np.linspace(x.min()-padx,x.max()+padx,grid_size); gy=np.linspace(y.min()-pady,y.max()+pady,grid_size)
    Xg,Yg=np.meshgrid(gx,gy); pos=np.vstack([Xg.ravel(),Yg.ravel()])
    cell=(gx[1]-gx[0])*(gy[1]-gy[0])
    out=[]
    for ids in parts:
        xx=x[ids]; yy=y[ids]
        try:
            k=gaussian_kde(np.vstack([xx,yy]),bw_method=bw_method)
            Z=k(pos).reshape(grid_size,grid_size)
        except Exception:
            return None
        m=Z*cell; m=m/m.sum(); Z=Z/m.sum()
        im=np.unravel_index(np.argmax(Z),Z.shape)
        order=np.argsort(Z.ravel())[::-1]; cum=np.cumsum(m.ravel()[order]); kk=np.searchsorted(cum,.90)
        thr=Z.ravel()[order[min(kk,len(order)-1)]]; area90=float(np.sum(Z>=thr)*cell)
        out.append({'mass':m,'Z':Z,'mode':(float(Xg[im]),float(Yg[im])),'area90':area90})
    m0,m1,m2=out
    overlap_early_late=float(np.minimum(m0['mass'],m2['mass']).sum())
    drift=float(np.hypot(m2['mode'][0]-m0['mode'][0],m2['mode'][1]-m0['mode'][1]))
    return {
        'KDE dinâmica - deslocamento do modo inicial→final (px)':drift,
        'KDE dinâmica - mudança da área HDR90 final-inicial (px²)':m2['area90']-m0['area90'],
        'KDE dinâmica - razão área HDR90 final/inicial':m2['area90']/m0['area90'] if m0['area90']>EPS else np.nan,
        'KDE dinâmica - sobreposição inicial-final':overlap_early_late,
    }, out, gx, gy

def compute_metrics(df, time_unit='ms', grid_n=8, target_x=None, target_y=None, target_radius=None, kde_grid=180, kde_bw='scott'):
    t_raw = df['TEMPO'].to_numpy(float)
    t_s = t_raw / 1000.0 if time_unit == 'ms' else t_raw.copy()
    x = df['X'].to_numpy(float)
    y = df['Y'].to_numpy(float)
    area = df['AREA'].to_numpy(float)

    dt = np.diff(t_s)
    dx = np.diff(x); dy = np.diff(y)
    vectors = np.column_stack([dx, dy])
    dist = np.hypot(dx, dy)
    valid_dt = dt > 0
    inst_speed = np.divide(dist, dt, out=np.full_like(dist, np.nan), where=valid_dt)
    angles = np.arctan2(dy, dx)
    dangle = np.angle(np.exp(1j*np.diff(angles))) if len(angles) > 1 else np.array([])

    duration = t_s[-1] - t_s[0] if len(t_s) > 1 else 0.0
    elapsed = t_s - t_s[0]
    weighted = vector_anisotropy(vectors, normalize=False)
    unweighted = vector_anisotropy(vectors, normalize=True)
    pell = point_ellipse_metrics(x, y)
    cmean, rlen, cvar = circular_metrics(angles)
    ah, ahn = angular_entropy(angles)
    seh, sehn = spatial_grid_entropy(x, y, grid_n=grid_n)

    if len(dt) >= 2:
        rmssd = float(np.sqrt(np.mean(np.diff(dt)**2)))
    else:
        rmssd = np.nan

    # Convex hull
    hull_area = hull_perim = np.nan
    if len(df) >= 3:
        try:
            hull = ConvexHull(np.column_stack([x, y]))
            hull_area = float(hull.volume)   # area in 2D
            hull_perim = float(hull.area)   # perimeter in 2D
        except Exception:
            pass

    total_dist = float(np.nansum(dist))
    net_disp = float(np.hypot(x[-1]-x[0], y[-1]-y[0])) if len(x)>1 else 0.0
    path_eff = float(net_disp/total_dist) if total_dist > EPS else np.nan
    cx, cy = float(np.mean(x)), float(np.mean(y))
    radial = np.hypot(x-cx, y-cy)
    kde_res = kde2d_metrics(x, y, grid_size=kde_grid, bw_method=kde_bw, target_x=target_x, target_y=target_y, target_radius=target_radius)
    dyn_kde = temporal_kde_comparison(df, grid_size=min(kde_grid,160), bw_method=kde_bw)

    metrics = {
        'Número de toques': len(df),
        'Duração do teste (s)': duration,
        'Frequência média de tapping (toques/s)': (len(df)-1)/duration if duration>0 and len(df)>1 else np.nan,
        'Intervalo médio entre toques (ms)': np.mean(dt)*1000 if len(dt) else np.nan,
        'Mediana do intervalo (ms)': np.median(dt)*1000 if len(dt) else np.nan,
        'DP do intervalo (ms)': np.std(dt, ddof=1)*1000 if len(dt)>1 else np.nan,
        'CV do intervalo (%)': safe_cv(dt),
        'IQR do intervalo (ms)': (np.percentile(dt,75)-np.percentile(dt,25))*1000 if len(dt) else np.nan,
        'RMSSD do intervalo (ms)': rmssd*1000 if np.isfinite(rmssd) else np.nan,
        'Autocorrelação lag-1 dos intervalos': lag1_autocorr(dt),
        'Inclinação do intervalo ao longo do teste (ms/s)': linear_slope(t_s[1:]-t_s[0], dt)*1000 if len(dt) else np.nan,
        'Entropia de Shannon temporal (bits)': shannon_entropy(dt),
        'Entropia de Shannon temporal normalizada': normalized_shannon(dt),
        'Sample entropy dos intervalos': sample_entropy(dt),
        'Distância média entre toques (px)': np.mean(dist) if len(dist) else np.nan,
        'Mediana da distância entre toques (px)': np.median(dist) if len(dist) else np.nan,
        'DP da distância entre toques (px)': np.std(dist, ddof=1) if len(dist)>1 else np.nan,
        'CV da distância entre toques (%)': safe_cv(dist),
        'Distância total percorrida (px)': total_dist,
        'Deslocamento líquido início-fim (px)': net_disp,
        'Eficiência da trajetória (deslocamento/distância)': path_eff,
        'Velocidade espacial média (px/s)': np.nanmean(inst_speed) if np.any(np.isfinite(inst_speed)) else np.nan,
        'DP da velocidade espacial (px/s)': np.nanstd(inst_speed, ddof=1) if np.sum(np.isfinite(inst_speed))>1 else np.nan,
        'X médio (px)': cx,
        'Y médio (px)': cy,
        'DP de X (px)': np.std(x, ddof=1) if len(x)>1 else np.nan,
        'DP de Y (px)': np.std(y, ddof=1) if len(y)>1 else np.nan,
        'RMS radial ao centróide (px)': np.sqrt(np.mean(radial**2)),
        'Raio mediano ao centróide (px)': np.median(radial),
        'Raio 95% ao centróide (px)': np.percentile(radial,95),
        'Área do fecho convexo (px²)': hull_area,
        'Perímetro do fecho convexo (px)': hull_perim,
        'Elipse espacial 95% - semi-eixo maior (px)': pell['major'],
        'Elipse espacial 95% - semi-eixo menor (px)': pell['minor'],
        'Elipse espacial 95% - razão maior/menor': pell['ratio'],
        'Elipse espacial 95% - excentricidade': pell['eccentricity'],
        'Elipse espacial 95% - ângulo (graus)': pell['angle'],
        'Elipse espacial 95% - área (px²)': pell['area95'],
        'Índice orientacional ponderado (razão dos semi-eixos)': weighted['ratio'],
        'Orientação ponderada - ângulo principal (graus)': weighted['angle'],
        'Índice orientacional não ponderado (razão dos semi-eixos)': unweighted['ratio'],
        'Orientação não ponderada - ângulo principal (graus)': unweighted['angle'],
        'Direção média dos vetores (graus)': cmean,
        'Comprimento resultante médio direcional R': rlen,
        'Variância circular das direções': cvar,
        'Entropia angular (bits)': ah,
        'Entropia angular normalizada': ahn,
        'Média |mudança angular| entre vetores (graus)': np.degrees(np.mean(np.abs(dangle))) if len(dangle) else np.nan,
        'DP da mudança angular (graus)': np.degrees(np.std(dangle, ddof=1)) if len(dangle)>1 else np.nan,
        'Entropia espacial em grade (bits)': seh,
        'Entropia espacial normalizada': sehn,
        'Proporção de toques na área alvo (%)': 100*np.nanmean(area==1) if len(area) else np.nan,
        'Número de acertos (AREA=1)': int(np.nansum(area==1)),
        'Número de erros (AREA≠1)': int(np.nansum(area!=1)),
        'Drift X (px/s)': linear_slope(elapsed, x),
        'Drift Y (px/s)': linear_slope(elapsed, y),
        'Drift radial do centróide (px/s)': linear_slope(elapsed, radial),
    }

    if kde_res is not None:
        metrics.update(kde_res['metrics'])
    if dyn_kde is not None:
        metrics.update(dyn_kde[0])

    if target_x is not None and target_y is not None:
        err = np.hypot(x-target_x, y-target_y)
        metrics.update({
            'Erro médio ao centro do alvo (px)': float(np.mean(err)),
            'Erro mediano ao centro do alvo (px)': float(np.median(err)),
            'RMSE ao centro do alvo (px)': float(np.sqrt(np.mean(err**2))),
            'Erro 95% ao centro do alvo (px)': float(np.percentile(err,95)),
            'Bias X em relação ao alvo (px)': float(np.mean(x-target_x)),
            'Bias Y em relação ao alvo (px)': float(np.mean(y-target_y)),
        })

    step_df = pd.DataFrame({
        'toque_destino': np.arange(2, len(df)+1),
        'tempo_s': t_s[1:],
        'intervalo_ms': dt*1000,
        'dx_px': dx,
        'dy_px': dy,
        'distancia_px': dist,
        'velocidade_px_s': inst_speed,
        'angulo_graus': np.degrees(angles) % 360,
    })
    return metrics, step_df, weighted, unweighted, kde_res, dyn_kde


def ellipse_trace_from_anisotropy(result, scale=2.0, n=240):
    if result['eigvals'] is None or result['eigvecs'] is None:
        return None, None
    vals = np.maximum(result['eigvals'], 0)
    vecs = result['eigvecs']
    th = np.linspace(0, 2*np.pi, n)
    circle = np.vstack([np.cos(th), np.sin(th)])
    axes = scale*np.sqrt(vals)[:,None]
    pts = vecs @ (axes*circle)
    return pts[0], pts[1]


def window_metrics(df, time_unit, window_s=5.0):
    t = df['TEMPO'].to_numpy(float) / (1000 if time_unit=='ms' else 1)
    t = t - t[0]
    if len(t)<2:
        return pd.DataFrame()
    max_t = t[-1]
    edges = np.arange(0, max_t + window_s, window_s)
    if len(edges)<2: edges=np.array([0,max_t])
    rows=[]
    for a,b in zip(edges[:-1],edges[1:]):
        mask=(t>=a)&(t<(b if b<max_t else b+EPS))
        idx=np.where(mask)[0]
        if len(idx)==0: continue
        sub=df.iloc[idx].copy()
        # intervals only inside window
        ts=sub['TEMPO'].to_numpy(float)/(1000 if time_unit=='ms' else 1)
        dts=np.diff(ts)
        xs=sub['X'].to_numpy(float); ys=sub['Y'].to_numpy(float)
        d=np.hypot(np.diff(xs),np.diff(ys)) if len(sub)>1 else np.array([])
        rows.append({
            'janela_inicio_s':a,
            'janela_fim_s':min(b,max_t),
            'n_toques':len(sub),
            'frequencia_toques_s': (len(sub)-1)/(ts[-1]-ts[0]) if len(sub)>1 and ts[-1]>ts[0] else np.nan,
            'iti_medio_ms':np.mean(dts)*1000 if len(dts) else np.nan,
            'iti_dp_ms':np.std(dts,ddof=1)*1000 if len(dts)>1 else np.nan,
            'distancia_media_px':np.mean(d) if len(d) else np.nan,
            'x_medio':np.mean(xs), 'y_medio':np.mean(ys)
        })
    return pd.DataFrame(rows)


# -------------------------
# UI
# -------------------------
st.title('Finger Tapping Test — análise quantitativa')
st.caption('Tempo + coordenadas X/Y + indicador de toque na área-alvo. A análise usa sempre as quatro primeiras colunas do arquivo.')

uploaded = st.file_uploader('Carregue o arquivo TXT ou CSV do FTT', type=['txt','csv'])

with st.sidebar:
    st.header('Configurações')
    time_unit = st.radio('Unidade da coluna de tempo', ['ms','s'], index=0, horizontal=True)
    grid_n = st.slider('Grade da entropia espacial', 3, 20, 8)
    window_s = st.slider('Janela para análise dinâmica (s)', 2, 15, 5)
    st.subheader('Kernel Density Estimation (KDE)')
    kde_grid = st.slider('Resolução da grade KDE', 80, 300, 180, 20)
    kde_bw = st.selectbox('Bandwidth da KDE', ['scott','silverman'], index=0)
    st.divider()
    st.subheader('Centro do alvo (opcional)')
    use_target = st.checkbox('Informar coordenadas do centro do alvo')
    target_x = st.number_input('X do alvo (px)', value=0.0) if use_target else None
    target_y = st.number_input('Y do alvo (px)', value=0.0) if use_target else None
    target_radius = st.number_input('Raio do alvo (px, opcional)', min_value=0.0, value=0.0, step=1.0) if use_target else 0.0
    st.caption('Se o centro real do alvo for fornecido, o app calcula erro, RMSE e bias espacial.')

if uploaded is None:
    st.info('Carregue um arquivo para iniciar a análise.')
    st.stop()

try:
    df = read_ftt_file(uploaded)
except Exception as e:
    st.error(str(e)); st.stop()

if len(df) < 3:
    st.error('São necessários pelo menos 3 eventos de toque para a maior parte das análises.')
    st.stop()

metrics, steps, weighted, unweighted, kde_res, dyn_kde = compute_metrics(
    df, time_unit=time_unit, grid_n=grid_n,
    target_x=target_x if use_target else None,
    target_y=target_y if use_target else None,
    target_radius=target_radius if use_target and target_radius>0 else None,
    kde_grid=kde_grid, kde_bw=kde_bw
)

# Summary cards
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric('Toques', f"{metrics['Número de toques']}")
c2.metric('Frequência', f"{metrics['Frequência média de tapping (toques/s)']:.2f} /s")
c3.metric('ITI médio', f"{metrics['Intervalo médio entre toques (ms)']:.1f} ms")
c4.metric('Distância total', f"{metrics['Distância total percorrida (px)']:.0f} px")
c5.metric('Acertos', f"{metrics['Proporção de toques na área alvo (%)']:.1f}%")

# Metrics table
st.subheader('Todas as métricas quantitativas')
metric_df = pd.DataFrame({'Variável': list(metrics.keys()), 'Valor': list(metrics.values())})
metric_df['Valor'] = metric_df['Valor'].apply(lambda v: np.nan if isinstance(v,float) and not np.isfinite(v) else v)
st.dataframe(metric_df, use_container_width=True, hide_index=True, height=620)

csv_metrics = metric_df.to_csv(index=False).encode('utf-8-sig')
st.download_button('Baixar métricas (.csv)', csv_metrics, 'FTT_metricas.csv', 'text/csv')

st.divider()
st.header('Visualizações')

tabs = st.tabs(['Trajetória', 'KDE espacial', 'Vetores orientacionais', 'Tempo', 'Distribuições', 'Dinâmica por janelas', 'Dados'])

with tabs[0]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['X'], y=df['Y'], mode='lines+markers',
                             marker=dict(size=6, color=np.arange(len(df)), colorscale='Viridis', showscale=True,
                                         colorbar=dict(title='Ordem')),
                             line=dict(width=1), name='Toques'))
    fig.add_trace(go.Scatter(x=[df['X'].iloc[0]], y=[df['Y'].iloc[0]], mode='markers',
                             marker=dict(size=13, symbol='circle-open'), name='Início'))
    fig.add_trace(go.Scatter(x=[df['X'].iloc[-1]], y=[df['Y'].iloc[-1]], mode='markers',
                             marker=dict(size=13, symbol='x'), name='Fim'))
    if use_target:
        fig.add_trace(go.Scatter(x=[target_x], y=[target_y], mode='markers',
                                 marker=dict(size=15, symbol='cross'), name='Centro alvo'))
    fig.update_layout(title='Sequência espacial dos toques', xaxis_title='X (px)', yaxis_title='Y (px)',
                      yaxis=dict(scaleanchor='x', scaleratio=1), height=650)
    st.plotly_chart(fig, use_container_width=True)

    # Density heatmap
    fig2 = px.density_heatmap(df, x='X', y='Y', nbinsx=30, nbinsy=30,
                              title='Densidade espacial de toques')
    fig2.update_yaxes(scaleanchor='x', scaleratio=1)
    st.plotly_chart(fig2, use_container_width=True)


with tabs[1]:
    st.subheader('Kernel Density Estimation 2D')
    if kde_res is None:
        st.warning('Não foi possível calcular a KDE para este conjunto de pontos.')
    else:
        figk = go.Figure(data=go.Contour(
            x=kde_res['gx'], y=kde_res['gy'], z=kde_res['Z'],
            contours=dict(coloring='heatmap', showlabels=True),
            colorbar=dict(title='Densidade')
        ))
        figk.add_trace(go.Scatter(x=df['X'], y=df['Y'], mode='markers',
                                  marker=dict(size=4, opacity=.35), name='Toques'))
        mx,my=kde_res['mode']
        figk.add_trace(go.Scatter(x=[mx],y=[my],mode='markers',marker=dict(size=13,symbol='x'),name='Modo KDE'))
        if use_target:
            figk.add_trace(go.Scatter(x=[target_x],y=[target_y],mode='markers',marker=dict(size=13,symbol='cross'),name='Centro alvo'))
            if target_radius>0:
                th=np.linspace(0,2*np.pi,240)
                figk.add_trace(go.Scatter(x=target_x+target_radius*np.cos(th), y=target_y+target_radius*np.sin(th),
                                          mode='lines', name='Limite do alvo'))
        figk.update_layout(title='KDE gaussiana 2D dos locais de toque',xaxis_title='X (px)',yaxis_title='Y (px)',
                           yaxis=dict(scaleanchor='x',scaleratio=1),height=650)
        st.plotly_chart(figk,use_container_width=True)

        rows=[]
        for lev,thr in kde_res['thresholds'].items():
            area=metrics.get(f'KDE - área HDR {int(lev*100)}% (px²)',np.nan)
            rows.append({'HDR':f'{int(lev*100)}%','Limiar de densidade':thr,'Área (px²)':area})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        st.caption('HDR = Highest Density Region: menor região da KDE que contém a fração indicada da massa de probabilidade.')

        if dyn_kde is not None:
            st.subheader('Mudança da densidade ao longo do teste')
            dyn_metrics,parts,gx,gy=dyn_kde
            labels=['Inicial','Intermediário','Final']
            figd=go.Figure()
            for lab,part in zip(labels,parts):
                figd.add_trace(go.Contour(x=gx,y=gy,z=part['Z'],showscale=False,contours=dict(coloring='lines'),name=lab,opacity=.75))
            figd.update_layout(title='Contornos KDE — primeiro, segundo e terceiro terços',xaxis_title='X (px)',yaxis_title='Y (px)',
                               yaxis=dict(scaleanchor='x',scaleratio=1),height=600)
            st.plotly_chart(figd,use_container_width=True)
            st.dataframe(pd.DataFrame({'Variável':list(dyn_metrics.keys()),'Valor':list(dyn_metrics.values())}),
                         use_container_width=True,hide_index=True)

with tabs[2]:
    dx = np.diff(df['X'].to_numpy(float)); dy = np.diff(df['Y'].to_numpy(float))
    norm = np.hypot(dx,dy)
    keep = norm > EPS
    U = np.column_stack([dx[keep],dy[keep]])
    Un = U / np.linalg.norm(U,axis=1,keepdims=True)

    colA,colB = st.columns(2)
    with colA:
        figw=go.Figure()
        for vx,vy in U:
            figw.add_trace(go.Scatter(x=[0,vx], y=[0,vy], mode='lines', line=dict(width=1), showlegend=False, opacity=.35))
        ex,ey=ellipse_trace_from_anisotropy(weighted, scale=2.0)
        if ex is not None:
            figw.add_trace(go.Scatter(x=ex,y=ey,mode='lines',name='Elipse'))
        figw.update_layout(title=f"Ponderado pela magnitude — índice = {weighted['ratio']:.3f}",
                           xaxis_title='ΔX', yaxis_title='ΔY', yaxis=dict(scaleanchor='x',scaleratio=1), height=560)
        st.plotly_chart(figw,use_container_width=True)

    with colB:
        fign=go.Figure()
        for vx,vy in Un:
            fign.add_trace(go.Scatter(x=[0,vx], y=[0,vy], mode='lines', line=dict(width=1), showlegend=False, opacity=.35))
        ex,ey=ellipse_trace_from_anisotropy(unweighted, scale=2.0)
        if ex is not None:
            fign.add_trace(go.Scatter(x=ex,y=ey,mode='lines',name='Elipse'))
        fign.update_layout(title=f"Somente direção (vetores unitários) — índice = {unweighted['ratio']:.3f}",
                           xaxis_title='ΔX normalizado', yaxis_title='ΔY normalizado',
                           yaxis=dict(scaleanchor='x',scaleratio=1), height=560)
        st.plotly_chart(fign,use_container_width=True)

    figpolar=go.Figure(go.Barpolar(theta=steps['angulo_graus'], r=steps['distancia_px'], opacity=.55))
    figpolar.update_layout(title='Distribuição polar dos vetores (raio = magnitude do deslocamento)', height=600)
    st.plotly_chart(figpolar,use_container_width=True)

with tabs[3]:
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=steps['tempo_s'], y=steps['intervalo_ms'], mode='lines+markers', name='ITI'))
    fig.update_layout(title='Intervalo entre toques ao longo do teste', xaxis_title='Tempo (s)', yaxis_title='ITI (ms)', height=450)
    st.plotly_chart(fig,use_container_width=True)

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=steps['tempo_s'], y=steps['distancia_px'], mode='lines+markers', name='Distância'))
    fig.update_layout(title='Distância entre toques ao longo do teste', xaxis_title='Tempo (s)', yaxis_title='Distância (px)', height=450)
    st.plotly_chart(fig,use_container_width=True)

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=steps['tempo_s'], y=steps['velocidade_px_s'], mode='lines+markers', name='Velocidade espacial'))
    fig.update_layout(title='Velocidade espacial entre eventos', xaxis_title='Tempo (s)', yaxis_title='px/s', height=450)
    st.plotly_chart(fig,use_container_width=True)

with tabs[4]:
    c1,c2=st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(steps,x='intervalo_ms',nbins=30,title='Distribuição dos intervalos'),use_container_width=True)
        st.plotly_chart(px.histogram(steps,x='angulo_graus',nbins=36,title='Distribuição angular'),use_container_width=True)
    with c2:
        st.plotly_chart(px.histogram(steps,x='distancia_px',nbins=30,title='Distribuição das distâncias'),use_container_width=True)
        st.plotly_chart(px.scatter(steps,x='intervalo_ms',y='distancia_px',trendline=None,
                                   title='Relação ITI × distância'),use_container_width=True)

with tabs[5]:
    win = window_metrics(df,time_unit,window_s)
    st.dataframe(win,use_container_width=True,hide_index=True)
    if not win.empty:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=win['janela_inicio_s'],y=win['frequencia_toques_s'],mode='lines+markers',name='Frequência'))
        fig.update_layout(title=f'Frequência por janelas de {window_s} s',xaxis_title='Início da janela (s)',yaxis_title='toques/s')
        st.plotly_chart(fig,use_container_width=True)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=win['janela_inicio_s'],y=win['iti_medio_ms'],mode='lines+markers',name='ITI'))
        fig.update_layout(title='ITI médio por janela',xaxis_title='Início da janela (s)',yaxis_title='ms')
        st.plotly_chart(fig,use_container_width=True)
        st.download_button('Baixar métricas por janela (.csv)', win.to_csv(index=False).encode('utf-8-sig'),
                           'FTT_metricas_janelas.csv','text/csv')

with tabs[6]:
    st.write('Eventos originais')
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.write('Variáveis derivadas entre toques consecutivos')
    st.dataframe(steps,use_container_width=True,hide_index=True)
    st.download_button('Baixar série derivada (.csv)',steps.to_csv(index=False).encode('utf-8-sig'),
                       'FTT_serie_derivada.csv','text/csv')

st.divider()
with st.expander('Como interpretar os dois índices orientacionais'):
    st.markdown('''
**Ponderado pela magnitude:** para cada par de toques sucessivos é formado o vetor `(ΔX, ΔY)`. Todos os vetores são transladados para a origem e calcula-se a matriz de segundo momento em torno da própria origem. Como vetores maiores contribuem quadraticamente mais, a elipse resultante incorpora **direção + magnitude**. O índice é a razão entre os semi-eixos principal e secundário.

**Não ponderado pela magnitude:** cada vetor é primeiro normalizado para comprimento 1. Depois aplica-se exatamente o mesmo procedimento. Assim, cada deslocamento tem o mesmo peso e o índice descreve predominantemente a **organização direcional**.

Nos dois casos, valor próximo de **1** indica distribuição aproximadamente isotrópica. Valores maiores indicam anisotropia/seletividade direcional crescente.
''')

with st.expander('Observações metodológicas'):
    st.markdown('''
- Distâncias estão em **pixels**, a menos que seja feita uma calibração física da tela.
- A coluna `AREA` é usada como classificação de acerto (`1`) versus erro (`≠1`).
- As entropias dependem da quantidade de dados e da discretização. Para comparações entre participantes, mantenha os mesmos parâmetros.
- A elipse espacial de 95% é calculada sobre as **posições dos toques**; os índices orientacionais são calculados sobre os **vetores entre toques** e representam conceitos diferentes.
- O centro do alvo não é inferido automaticamente, porque isso confundiria precisão observada com a localização real do estímulo. Quando conhecido, informe-o na barra lateral.
- A KDE usa kernel gaussiano bidimensional. As áreas HDR 50/75/90/95% representam as menores regiões que concentram essas proporções da massa estimada.
- A massa KDE dentro do alvo só é calculada quando centro e raio do alvo são informados.
- O número de picos da KDE depende do bandwidth; por isso, para comparações entre participantes, mantenha o mesmo método de bandwidth e a mesma resolução de grade.
''')
