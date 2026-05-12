from __future__ import annotations

import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


def _configure_qt_plugin_env() -> None:
    # Some mixed conda/pip setups resolve Qt plugin paths incorrectly.
    # Use a known-good default only when the user has not set one.
    if os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
        return

    exe_path = Path(sys.executable).resolve()
    env_root = exe_path.parent

    candidate_plugins = [
        env_root / "lib" / "site-packages" / "PySide6" / "plugins",
        env_root / "Lib" / "site-packages" / "PySide6" / "plugins",
    ]

    for plugins_dir in candidate_plugins:
        platform_dir = plugins_dir / "platforms"
        if (platform_dir / "qwindows.dll").exists():
            os.environ["QT_PLUGIN_PATH"] = str(plugins_dir)
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platform_dir)
            break


_configure_qt_plugin_env()

import numpy as np
import pandas as pd
import pytz
import utide
import xarray as xr
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, Qt, QUrl, Signal, Slot
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from scipy.spatial import cKDTree
from utide import ut_constants
from utide.utilities import Bunch

MODEL_FILE = "CSIRO_tidal_const_v12.nc"
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("prediction_outputs")

DEFAULT_CONSTITUENTS = [
    "Q1",
    "O1",
    "P1",
    "K1",
    "2N2",
    "N2",
    "M2",
    "S2",
    "K2",
    "M4",
    "MS4",
    "M6",
    "2MS6",
]

TIMEZONES = [
    "Australia/Brisbane",
    "Australia/Sydney",
    "Australia/Melbourne",
    "Australia/Hobart",
    "Australia/Adelaide",
    "Australia/Perth",
    "Australia/Darwin",
]

FREQUENCIES = ["10min", "30min", "1H", "3H", "6H", "12H", "1D"]

UTIDE_NAMES_UPPER = [str(name).strip().upper() for name in ut_constants["const"]["name"]]
UTIDE_NAME_TO_INDEX = {name: idx for idx, name in enumerate(UTIDE_NAMES_UPPER)}
UTIDE_NAME_SET = set(UTIDE_NAME_TO_INDEX.keys())
UTIDE_FREQ = ut_constants["const"]["freq"]


class CsiroTideModel:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"NetCDF file not found: {self.model_path}")

        self.ds = xr.open_dataset(self.model_path)
        self.face_lon = np.asarray(self.ds["Mesh2_face_x"].values, dtype=float)
        self.face_lat = np.asarray(self.ds["Mesh2_face_y"].values, dtype=float)
        self.tree = self._build_face_tree(self.face_lon, self.face_lat)
        self.constituents = self._read_constituent_names(self.ds)

    @staticmethod
    def _build_face_tree(lon_deg: np.ndarray, lat_deg: np.ndarray) -> cKDTree:
        lon = np.deg2rad(lon_deg)
        lat = np.deg2rad(lat_deg)
        xyz = np.column_stack(
            [
                np.cos(lat) * np.cos(lon),
                np.cos(lat) * np.sin(lon),
                np.sin(lat),
            ]
        )
        return cKDTree(xyz)

    @staticmethod
    def _query_face_tree(tree: cKDTree, lon_deg: float, lat_deg: float) -> tuple[int, float]:
        lon = np.deg2rad(float(lon_deg))
        lat = np.deg2rad(float(lat_deg))
        q = np.array([[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]])

        chord, idx = tree.query(q, k=1)
        theta = 2 * np.arcsin(np.clip(chord / 2, 0, 1))
        km = 6371.0 * theta
        return int(np.asarray(idx).item()), float(np.asarray(km).item())

    @staticmethod
    def _decode_constituent_array(values: np.ndarray) -> list[str]:
        arr = np.asarray(values)

        if arr.ndim == 1:
            names = [
                item.decode("utf-8").strip() if isinstance(item, (bytes, np.bytes_)) else str(item).strip()
                for item in arr
            ]
            return [name for name in names if name]

        if arr.ndim == 2:
            decoded: list[str] = []
            for row in arr:
                chars = [item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item) for item in row]
                decoded.append("".join(chars).strip())
            return [name for name in decoded if name]

        return []

    @staticmethod
    def _count_compatible_constituents(names: Iterable[str]) -> int:
        return sum(1 for name in names if str(name).strip().upper() in UTIDE_NAME_SET)

    def _read_constituent_names(self, ds: xr.Dataset) -> list[str]:
        for key in ["constituent_name", "constituent", "constituents"]:
            if key in ds.variables:
                names = self._decode_constituent_array(ds[key].values)
                if names and self._count_compatible_constituents(names) > 0:
                    return names
        return DEFAULT_CONSTITUENTS.copy()

    @staticmethod
    def _datetime_to_ordinal_float(ts: pd.Timestamp) -> float:
        ts = pd.Timestamp(ts)
        return float(ts.toordinal() + (ts.hour + ts.minute / 60 + ts.second / 3600) / 24.0)

    def _reconstruct_tides(self, iface: int, lat: float, times_utc: pd.DatetimeIndex) -> np.ndarray:
        amplitudes = np.asarray(self.ds["h_amp"].isel(nMesh2_face=iface).values, dtype=float)
        phases = np.asarray(self.ds["h_pha"].isel(nMesh2_face=iface).values, dtype=float)

        constituents = self.constituents
        if (len(constituents) != len(amplitudes)) or (self._count_compatible_constituents(constituents) == 0):
            constituents = DEFAULT_CONSTITUENTS[: len(amplitudes)]

        keep_idx: list[int] = []
        ut_idx: list[int] = []
        names_kept: list[str] = []

        for idx, name in enumerate(constituents):
            upper_name = str(name).strip().upper()
            ut_i = UTIDE_NAME_TO_INDEX.get(upper_name)
            if ut_i is not None:
                keep_idx.append(idx)
                ut_idx.append(ut_i)
                names_kept.append(upper_name)

        if not keep_idx:
            raise ValueError("No compatible harmonic constituents found.")

        a = amplitudes[np.asarray(keep_idx, dtype=int)]
        g = phases[np.asarray(keep_idx, dtype=int)]
        freq = UTIDE_FREQ[np.asarray(ut_idx, dtype=int)]

        reftime = self._datetime_to_ordinal_float(times_utc[0])

        coef = Bunch(name=names_kept, mean=0.0, slope=0.0)
        coef["A"] = np.asarray(a, dtype=float)
        coef["g"] = np.asarray(g, dtype=float)
        coef["A_ci"] = np.zeros_like(coef["A"])
        coef["g_ci"] = np.zeros_like(coef["g"])
        coef["aux"] = Bunch(reftime=reftime, lind=np.asarray(ut_idx, dtype=int), frq=freq, lat=float(lat))
        coef["aux"]["opt"] = Bunch(
            twodim=False,
            epoch="python",
            phase="Greenwich",
            nodal=True,
            trend=False,
            verbose=False,
            nodiagn=True,
            diagnminsnr=2,
            rmin=1,
            ordercnstit="PE",
            cnstit="auto",
            nodsatlint=False,
            nodsatnone=False,
            nodesatlint=False,
            nodesatnone=False,
            gwchlint=False,
            gwchnone=False,
            prefilt=[],
            white=False,
            RunTimeDisp=False,
            equi=False,
            infer=None,
            inferaprx=0,
            notrend=True,
            linci=True,
            lsfrqosmp=1,
            nrlzn=200,
            tunrdn=1,
        )

        out = utide.reconstruct(times_utc.to_pydatetime(), coef, verbose=False)
        return np.asarray(out.h, dtype=float)

    def predict_series(
        self,
        lon: float,
        lat: float,
        times_local: pd.DatetimeIndex,
        timezone_name: str,
    ) -> tuple[pd.Series, float]:
        iface, distance_km = self._query_face_tree(self.tree, lon, lat)

        tz = pytz.timezone(timezone_name)
        if times_local.tz is None:
            times_local = times_local.tz_localize(tz)
        else:
            times_local = times_local.tz_convert(tz)

        times_utc = times_local.tz_convert("UTC")
        tide_vals = self._reconstruct_tides(iface=iface, lat=lat, times_utc=times_utc)

        tide_series = pd.Series(tide_vals, index=times_local, name="tide_m")
        return tide_series, distance_km

    def predict_series_for_faces(
        self,
        face_indices: np.ndarray,
        times_local: pd.DatetimeIndex,
        timezone_name: str,
    ) -> pd.Series:
        face_indices = np.asarray(face_indices, dtype=int)
        if face_indices.size == 0:
            raise ValueError("No mesh faces provided")

        tz = pytz.timezone(timezone_name)
        if times_local.tz is None:
            times_local = times_local.tz_localize(tz)
        else:
            times_local = times_local.tz_convert(tz)

        times_utc = times_local.tz_convert("UTC")
        total = np.zeros(len(times_utc), dtype=float)

        for iface in face_indices:
            lat = float(self.face_lat[int(iface)])
            total += self._reconstruct_tides(iface=int(iface), lat=lat, times_utc=times_utc)

        mean_vals = total / float(face_indices.size)
        return pd.Series(mean_vals, index=times_local, name="tide_m")


def parse_datetime_flexible(value: str) -> pd.Timestamp:
    text = str(value).strip()

    fmts = [
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ]

    for fmt in fmts:
        try:
            return pd.Timestamp(pd.to_datetime(text, format=fmt))
        except ValueError:
            continue

    dt = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"Could not parse date/time: {value}")
    return pd.Timestamp(dt)


def parse_row_selection(selection: str, max_count: int) -> list[int]:
    text = selection.strip()
    if not text:
        return list(range(max_count))

    out: set[int] = set()
    for token in [t.strip() for t in text.split(",") if t.strip()]:
        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                raise ValueError("Invalid range format. Use forms like 1-5,7,9-12")
            start = int(parts[0])
            end = int(parts[1])
            if start <= 0 or end <= 0 or end < start:
                raise ValueError("Invalid row range values")
            for i in range(start, end + 1):
                idx = i - 1
                if 0 <= idx < max_count:
                    out.add(idx)
        else:
            if not token.isdigit():
                raise ValueError("Invalid row index. Use forms like 3 or 3-6")
            idx = int(token) - 1
            if 0 <= idx < max_count:
                out.add(idx)

    if not out:
        raise ValueError("No valid rows selected")
    return sorted(out)


def _parse_kml_polygons(path: str | Path) -> list[np.ndarray]:
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    polys: list[np.ndarray] = []
    for elem in root.findall(".//kml:Polygon", ns):
        coord_el = elem.find(".//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns)
        if coord_el is None or not coord_el.text:
            continue

        pts: list[tuple[float, float]] = []
        for token in coord_el.text.strip().split():
            parts = token.split(",")
            if len(parts) < 2:
                continue
            lon = float(parts[0])
            lat = float(parts[1])
            pts.append((lon, lat))

        if len(pts) >= 3:
            polys.append(np.asarray(pts, dtype=float))

    return polys


def _parse_shp_polygons(path: str | Path) -> list[np.ndarray]:
    try:
        import shapefile  # pyshp
    except Exception as exc:
        raise ImportError("SHP support requires 'pyshp'. Install with: pip install pyshp") from exc

    shp_path = Path(path)
    reader = shapefile.Reader(str(shp_path))
    polys: list[np.ndarray] = []

    for shape in reader.shapes():
        if shape.shapeType not in (shapefile.POLYGON, shapefile.POLYGONM, shapefile.POLYGONZ):
            continue
        points = shape.points
        parts = list(shape.parts) + [len(points)]
        for i in range(len(parts) - 1):
            ring = points[parts[i] : parts[i + 1]]
            if len(ring) >= 3:
                arr = np.asarray([(float(p[0]), float(p[1])) for p in ring], dtype=float)
                polys.append(arr)

    # Reproject to EPSG:4326 when .prj is available and not already geographic WGS84.
    prj_path = shp_path.with_suffix(".prj")
    if prj_path.exists():
        try:
            from pyproj import CRS, Transformer
        except Exception as exc:
            raise ImportError(
                "CRS transformation requires 'pyproj'. Install with: pip install pyproj"
            ) from exc

        src_wkt = prj_path.read_text(encoding="utf-8", errors="ignore")
        src_crs = CRS.from_wkt(src_wkt)
        dst_crs = CRS.from_epsg(4326)
        if src_crs != dst_crs:
            transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
            out: list[np.ndarray] = []
            for poly in polys:
                x = poly[:, 0]
                y = poly[:, 1]
                lon, lat = transformer.transform(x, y)
                out.append(np.column_stack([lon, lat]).astype(float))
            polys = out
    else:
        # If CRS metadata is missing, we can only accept data that already looks like lon/lat.
        if polys:
            all_x = np.concatenate([p[:, 0] for p in polys])
            all_y = np.concatenate([p[:, 1] for p in polys])
            looks_like_lonlat = (
                np.nanmin(all_x) >= -180
                and np.nanmax(all_x) <= 180
                and np.nanmin(all_y) >= -90
                and np.nanmax(all_y) <= 90
            )
            if not looks_like_lonlat:
                raise ValueError(
                    "Shapefile appears to be projected but has no .prj CRS file. "
                    "Provide a .prj file so geometry can be transformed to EPSG:4326."
                )

    return polys


def faces_in_polygons(face_lon: np.ndarray, face_lat: np.ndarray, polygons: list[np.ndarray]) -> np.ndarray:
    if not polygons:
        return np.array([], dtype=int)

    points = np.column_stack([face_lon, face_lat])
    mask = np.zeros(points.shape[0], dtype=bool)

    for poly in polygons:
        if poly.shape[0] < 3:
            continue
        path = np.asarray(poly, dtype=float)
        # Vectorized ray-casting using matplotlib's Path.
        from matplotlib.path import Path as MplPath

        mask |= MplPath(path).contains_points(points)

    return np.where(mask)[0]


class OSMTileView(QGraphicsView):
    pointChanged = Signal(float, float)

    TILE_SIZE = 256
    MIN_ZOOM = 2
    MAX_ZOOM = 18

    def __init__(
        self,
        lat: float,
        lon: float,
        zoom: int = 6,
        parent: QWidget | None = None,
        show_marker: bool = True,
    ):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self.center_lat = float(lat)
        self.center_lon = float(lon)
        self.zoom = int(max(self.MIN_ZOOM, min(self.MAX_ZOOM, zoom)))

        self._network = QNetworkAccessManager(self)
        self._network.finished.connect(self._on_tile_reply)
        self._pending: dict[QNetworkReply, tuple[int, int, int]] = {}
        self._inflight_keys: set[tuple[int, int, int]] = set()
        self._pix_cache: dict[tuple[int, int, int], QPixmap] = {}

        self._marker_lat = self.center_lat
        self._marker_lon = self.center_lon
        self._show_marker = bool(show_marker)
        self._overlay_polygons: list[np.ndarray] = []
        self._panning = False
        self._drag_moved = False
        self._press_pos = None
        self._press_center_world = None
        self._wheel_accum = 0

        self._placeholder = QPixmap(self.TILE_SIZE, self.TILE_SIZE)
        self._placeholder.fill(QColor("#e5eaf2"))

        self.setMouseTracking(True)

    @staticmethod
    def _clip_lat(lat: float) -> float:
        return max(-85.05112878, min(85.05112878, lat))

    def _lonlat_to_world(self, lat: float, lon: float, zoom: int) -> tuple[float, float]:
        lat = self._clip_lat(lat)
        world = self.TILE_SIZE * (2**zoom)
        x = (lon + 180.0) / 360.0 * world
        lat_rad = math.radians(lat)
        y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * world
        return x, y

    def _normalize_world_xy(self, x: float, y: float, zoom: int) -> tuple[float, float]:
        world = self.TILE_SIZE * (2**zoom)
        x = x % world
        y = max(0.0, min(world - 1.0, y))
        return x, y

    def _world_to_lonlat(self, x: float, y: float, zoom: int) -> tuple[float, float]:
        world = self.TILE_SIZE * (2**zoom)
        x, y = self._normalize_world_xy(x, y, zoom)
        lon = x / world * 360.0 - 180.0
        n = math.pi - (2.0 * math.pi * y) / world
        lat = math.degrees(math.atan(math.sinh(n)))
        return self._clip_lat(lat), lon

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_tiles()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._drag_moved = False
            self._press_pos = event.position()
            self._press_center_world = self._lonlat_to_world(self.center_lat, self.center_lon, self.zoom)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning and self._press_pos is not None and self._press_center_world is not None:
            dx = float(event.position().x() - self._press_pos.x())
            dy = float(event.position().y() - self._press_pos.y())
            if abs(dx) > 2 or abs(dy) > 2:
                self._drag_moved = True
            cx, cy = self._press_center_world
            new_x = cx - dx
            new_y = cy - dy
            new_x, new_y = self._normalize_world_xy(new_x, new_y, self.zoom)
            self.center_lat, self.center_lon = self._world_to_lonlat(new_x, new_y, self.zoom)
            self._render_tiles()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._press_pos is not None:
            if not self._drag_moved:
                lat, lon = self._screen_to_lonlat(event.position().x(), event.position().y())
                self._handle_click(lat, lon)
            self._panning = False
            self._press_pos = None
            self._press_center_world = None
        super().mouseReleaseEvent(event)

    def _handle_click(self, lat: float, lon: float) -> None:
        self._marker_lat = lat
        self._marker_lon = lon
        self.pointChanged.emit(lat, lon)
        self._render_tiles()

    def set_overlay_polygons(self, polygons: list[np.ndarray]) -> None:
        self._overlay_polygons = [np.asarray(p, dtype=float) for p in polygons]
        self._render_tiles()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        # Smooth wheel behavior (especially on touchpads) and only step one zoom level at a time.
        self._wheel_accum += int(delta)
        steps = int(self._wheel_accum / 120)
        if steps == 0:
            return
        self._wheel_accum -= steps * 120
        new_zoom = self.zoom + (1 if steps > 0 else -1)
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, new_zoom))
        if new_zoom == self.zoom:
            return

        cursor_x = float(event.position().x())
        cursor_y = float(event.position().y())
        before_x, before_y = self._screen_to_world(cursor_x, cursor_y, self.zoom)

        old_zoom = self.zoom
        self.zoom = new_zoom
        scale = 2 ** (new_zoom - old_zoom)
        after_x = before_x * scale
        after_y = before_y * scale

        vx = max(1.0, float(self.viewport().width()))
        vy = max(1.0, float(self.viewport().height()))
        center_x = after_x - cursor_x + vx / 2.0
        center_y = after_y - cursor_y + vy / 2.0
        center_x, center_y = self._normalize_world_xy(center_x, center_y, self.zoom)
        self.center_lat, self.center_lon = self._world_to_lonlat(center_x, center_y, self.zoom)
        self._render_tiles()
        event.accept()

    def _screen_to_world(self, sx: float, sy: float, zoom: int) -> tuple[float, float]:
        center_x, center_y = self._lonlat_to_world(self.center_lat, self.center_lon, zoom)
        ox = center_x - float(self.viewport().width()) / 2.0
        oy = center_y - float(self.viewport().height()) / 2.0
        return ox + sx, oy + sy

    def _screen_to_lonlat(self, sx: float, sy: float) -> tuple[float, float]:
        wx, wy = self._screen_to_world(sx, sy, self.zoom)
        return self._world_to_lonlat(wx, wy, self.zoom)

    def _lonlat_to_screen(self, lat: float, lon: float) -> tuple[float, float]:
        vw = max(1.0, float(self.viewport().width()))
        vh = max(1.0, float(self.viewport().height()))
        center_x, center_y = self._lonlat_to_world(self.center_lat, self.center_lon, self.zoom)
        origin_x = center_x - vw / 2.0
        origin_y = center_y - vh / 2.0
        world_x, world_y = self._lonlat_to_world(lat, lon, self.zoom)
        return world_x - origin_x, world_y - origin_y

    def _request_tile(self, z: int, x: int, y: int) -> None:
        key = (z, x, y)
        if key in self._pix_cache or key in self._inflight_keys:
            return
        url = QUrl(f"https://tile.openstreetmap.org/{z}/{x}/{y}.png")
        req = QNetworkRequest(url)
        req.setHeader(QNetworkRequest.KnownHeaders.UserAgentHeader, "TidePredictionTool/1.0 (PySide6)")
        reply = self._network.get(req)
        self._pending[reply] = key
        self._inflight_keys.add(key)

    def _on_tile_reply(self, reply: QNetworkReply) -> None:
        key = self._pending.pop(reply, None)
        if key is None:
            reply.deleteLater()
            return

        self._inflight_keys.discard(key)

        if reply.error() == QNetworkReply.NetworkError.NoError:
            data = bytes(reply.readAll())
            pix = QPixmap()
            if pix.loadFromData(data):
                self._pix_cache[key] = pix
        reply.deleteLater()
        self._render_tiles()

    def _render_tiles(self) -> None:
        scene = self.scene()
        if scene is None:
            return

        vw = max(1, self.viewport().width())
        vh = max(1, self.viewport().height())
        scene.clear()
        scene.setSceneRect(0, 0, vw, vh)

        center_x, center_y = self._lonlat_to_world(self.center_lat, self.center_lon, self.zoom)
        origin_x = center_x - vw / 2.0
        origin_y = center_y - vh / 2.0

        tiles_per_axis = 2**self.zoom
        x_start = int(math.floor(origin_x / self.TILE_SIZE)) - 1
        x_end = int(math.floor((origin_x + vw) / self.TILE_SIZE)) + 1
        y_start = int(math.floor(origin_y / self.TILE_SIZE)) - 1
        y_end = int(math.floor((origin_y + vh) / self.TILE_SIZE)) + 1

        for tx in range(x_start, x_end + 1):
            wrap_x = tx % tiles_per_axis
            screen_x = tx * self.TILE_SIZE - origin_x
            for ty in range(y_start, y_end + 1):
                if ty < 0 or ty >= tiles_per_axis:
                    continue
                screen_y = ty * self.TILE_SIZE - origin_y
                cache_key = (self.zoom, wrap_x, ty)
                pix = self._pix_cache.get(cache_key)
                if pix is None:
                    self._request_tile(self.zoom, wrap_x, ty)
                    pix = self._placeholder
                item = QGraphicsPixmapItem(pix)
                item.setPos(screen_x, screen_y)
                scene.addItem(item)

        for poly in self._overlay_polygons:
            if poly.shape[0] < 2:
                continue
            points = [self._lonlat_to_screen(lat=float(p[1]), lon=float(p[0])) for p in poly]
            for i in range(len(points) - 1):
                line = QGraphicsLineItem(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
                line.setPen(QPen(QColor("#ff9f1a"), 2))
                line.setZValue(6)
                scene.addItem(line)
            if len(points) >= 3:
                line = QGraphicsLineItem(points[-1][0], points[-1][1], points[0][0], points[0][1])
                line.setPen(QPen(QColor("#ff9f1a"), 2))
                line.setZValue(6)
                scene.addItem(line)

        if self._show_marker:
            marker_x, marker_y = self._lonlat_to_world(self._marker_lat, self._marker_lon, self.zoom)
            marker_sx = marker_x - origin_x
            marker_sy = marker_y - origin_y
            marker = QGraphicsEllipseItem(marker_sx - 6.0, marker_sy - 6.0, 12.0, 12.0)
            marker.setPen(QPen(QColor("#a5091f"), 2))
            marker.setBrush(QColor("#ff4a64"))
            marker.setZValue(7)
            scene.addItem(marker)


class PolygonDrawMapView(OSMTileView):
    polygonChanged = Signal()

    def __init__(self, lat: float, lon: float, zoom: int = 6, parent: QWidget | None = None):
        super().__init__(lat=lat, lon=lon, zoom=zoom, parent=parent, show_marker=False)
        self.vertices: list[tuple[float, float]] = []  # (lon, lat)

    def _handle_click(self, lat: float, lon: float) -> None:
        self.vertices.append((float(lon), float(lat)))
        self.set_overlay_polygons([np.asarray(self.vertices, dtype=float)])
        self.polygonChanged.emit()

    def undo_last_vertex(self) -> None:
        if self.vertices:
            self.vertices.pop()
            self.set_overlay_polygons([np.asarray(self.vertices, dtype=float)] if self.vertices else [])
            self.polygonChanged.emit()

    def clear_vertices(self) -> None:
        self.vertices = []
        self.set_overlay_polygons([])
        self.polygonChanged.emit()

    def polygon_array(self) -> np.ndarray | None:
        if len(self.vertices) < 3:
            return None
        return np.asarray(self.vertices, dtype=float)


class PolygonDrawDialog(QDialog):
    def __init__(self, center_lat: float = -23.0, center_lon: float = 150.0, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Draw Survey Polygon")
        self.resize(980, 720)
        self.polygon: np.ndarray | None = None

        layout = QVBoxLayout(self)
        info = QLabel("Click to add polygon vertices. Drag to pan. Mouse wheel to zoom.")
        self.count_label = QLabel("Vertices: 0")

        self.map_view = PolygonDrawMapView(center_lat, center_lon, zoom=6, parent=self)
        self.map_view.polygonChanged.connect(self._update_vertex_count)

        actions = QHBoxLayout()
        self.undo_btn = QPushButton("Undo Last Vertex")
        self.undo_btn.setProperty("variant", "secondary")
        self.undo_btn.clicked.connect(self.map_view.undo_last_vertex)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setProperty("variant", "secondary")
        self.clear_btn.clicked.connect(self.map_view.clear_vertices)
        actions.addWidget(self.undo_btn)
        actions.addWidget(self.clear_btn)
        actions.addStretch(1)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._accept_if_valid)
        button_box.rejected.connect(self.reject)

        layout.addWidget(info)
        layout.addWidget(self.count_label)
        layout.addLayout(actions)
        layout.addWidget(self.map_view, stretch=1)
        layout.addWidget(button_box)

    def _update_vertex_count(self) -> None:
        self.count_label.setText(f"Vertices: {len(self.map_view.vertices)}")

    def _accept_if_valid(self) -> None:
        poly = self.map_view.polygon_array()
        if poly is None:
            QMessageBox.warning(self, "Invalid Polygon", "Add at least 3 vertices to define a polygon.")
            return
        self.polygon = poly
        self.accept()


class MapPickerDialog(QDialog):
    def __init__(self, initial_lat: float = -23.0, initial_lon: float = 150.0, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Select Point on Map")
        self.resize(920, 680)

        self.selected_lat = float(initial_lat)
        self.selected_lon = float(initial_lon)

        layout = QVBoxLayout(self)

        self.help_label = QLabel("Pan with drag, zoom with mouse wheel, click to select location.")
        self.coords_label = QLabel(self._format_coords(self.selected_lat, self.selected_lon))

        self.map_view = OSMTileView(self.selected_lat, self.selected_lon, zoom=6, parent=self)
        self.map_view.pointChanged.connect(self._handle_point_changed)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout.addWidget(self.help_label)
        layout.addWidget(self.coords_label)
        layout.addWidget(self.map_view, stretch=1)
        layout.addWidget(button_box)

    @staticmethod
    def _format_coords(lat: float, lon: float) -> str:
        return f"Latitude: {lat:.6f} | Longitude: {lon:.6f}"

    def _handle_point_changed(self, lat: float, lon: float) -> None:
        self.selected_lat = float(lat)
        self.selected_lon = float(lon)
        self.coords_label.setText(self._format_coords(self.selected_lat, self.selected_lon))


class TidePredictionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TidePredictionApp")
        self.resize(1320, 860)

        INPUT_DIR.mkdir(exist_ok=True)
        OUTPUT_DIR.mkdir(exist_ok=True)

        self.model = CsiroTideModel(MODEL_FILE)
        self.single_results: pd.Series | None = None
        self.single_distance_km: float | None = None
        self.single_site_name = "site"
        self.survey_results: pd.Series | None = None
        self.survey_name = "survey"
        self.survey_polygons: list[np.ndarray] = []
        self.survey_face_indices: np.ndarray = np.array([], dtype=int)
        self.batch_df: pd.DataFrame | None = None

        self._build_ui()
        self._apply_stylesheet()
        self._apply_button_icons()

    def _build_ui(self) -> None:
        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(16, 14, 16, 14)
        root_layout.setSpacing(10)
        self.setCentralWidget(central)

        header_card = QGroupBox()
        header_card.setObjectName("headerCard")
        header_layout = QVBoxLayout(header_card)
        header_layout.setContentsMargins(16, 10, 16, 12)
        header_layout.setSpacing(2)

        title_label = QLabel("TidePredictionApp")
        title_label.setObjectName("headerTitle")
        subtitle_label = QLabel("CSIRO tidal prediction with single-site and batch workflows")
        subtitle_label.setObjectName("headerSubtitle")
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        root_layout.addWidget(header_card)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        root_layout.addWidget(self.tabs)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusBarLabel")
        root_layout.addWidget(self.status_label)

        self.tabs.addTab(self._build_single_tab(), "Single Location")
        self.tabs.addTab(self._build_survey_tab(), "Survey Location")
        self.tabs.addTab(self._build_batch_tab(), "Batch Processing")

    def _build_survey_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        form_box = QGroupBox("Survey Area Inputs")
        form_layout = QGridLayout(form_box)

        self.survey_name_edit = QLineEdit("survey")
        self.survey_start_edit = QLineEdit()
        self.survey_end_edit = QLineEdit()
        self.survey_freq_combo = QComboBox()
        self.survey_freq_combo.addItems(FREQUENCIES)
        self.survey_freq_combo.setCurrentText("1H")
        self.survey_tz_combo = QComboBox()
        self.survey_tz_combo.addItems(TIMEZONES)
        self.survey_tz_combo.setCurrentText("Australia/Brisbane")

        self.survey_polygon_file_edit = QLineEdit()
        self.survey_browse_btn = QPushButton("Browse KML/SHP")
        self.survey_browse_btn.setProperty("variant", "secondary")
        self.survey_browse_btn.clicked.connect(self.load_survey_polygon_file)
        self.survey_draw_btn = QPushButton("Draw Polygon on Map")
        self.survey_draw_btn.setProperty("variant", "secondary")
        self.survey_draw_btn.clicked.connect(self.draw_survey_polygon)
        self.survey_clear_polygon_btn = QPushButton("Clear Polygon")
        self.survey_clear_polygon_btn.setProperty("variant", "secondary")
        self.survey_clear_polygon_btn.clicked.connect(self.clear_survey_polygon)

        self.survey_face_info_label = QLabel("Survey polygon: none")
        self.survey_face_info_label.setObjectName("infoPill")

        form_layout.addWidget(QLabel("Survey Name"), 0, 0)
        form_layout.addWidget(self.survey_name_edit, 0, 1)
        form_layout.addWidget(QLabel("Timezone"), 0, 2)
        form_layout.addWidget(self.survey_tz_combo, 0, 3)

        form_layout.addWidget(QLabel("Start Date (YYYY-MM-DD)"), 1, 0)
        form_layout.addWidget(self.survey_start_edit, 1, 1)
        form_layout.addWidget(QLabel("End Date (YYYY-MM-DD)"), 1, 2)
        form_layout.addWidget(self.survey_end_edit, 1, 3)

        form_layout.addWidget(QLabel("Frequency"), 2, 0)
        form_layout.addWidget(self.survey_freq_combo, 2, 1)
        form_layout.addWidget(QLabel("Polygon File"), 2, 2)
        form_layout.addWidget(self.survey_polygon_file_edit, 2, 3)

        form_layout.addWidget(self.survey_browse_btn, 3, 0)
        form_layout.addWidget(self.survey_draw_btn, 3, 1)
        form_layout.addWidget(self.survey_clear_polygon_btn, 3, 2)
        form_layout.addWidget(self.survey_face_info_label, 3, 3)

        self.survey_run_btn = QPushButton("Run Survey Analysis")
        self.survey_run_btn.setProperty("variant", "primary")
        self.survey_run_btn.clicked.connect(self.run_survey_analysis)

        self.survey_save_csv_btn = QPushButton("Save Survey CSV")
        self.survey_save_csv_btn.setProperty("variant", "secondary")
        self.survey_save_csv_btn.clicked.connect(self.save_survey_csv)

        action_row = QHBoxLayout()
        action_row.addWidget(self.survey_run_btn)
        action_row.addWidget(self.survey_save_csv_btn)
        action_row.addStretch(1)

        self.survey_figure = Figure(figsize=(10, 4), dpi=100)
        self.survey_ax = self.survey_figure.add_subplot(111)
        self.survey_ax.set_title("Survey Area Average Tide Prediction")
        self.survey_ax.set_xlabel("Time")
        self.survey_ax.set_ylabel("Tide (m)")
        self.survey_ax.grid(alpha=0.25)
        self.survey_canvas = FigureCanvasQTAgg(self.survey_figure)

        layout.addWidget(form_box)
        layout.addLayout(action_row)
        layout.addWidget(self.survey_canvas, stretch=1)

        return panel

    def _build_single_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        form_box = QGroupBox("Single Site Inputs")
        form_layout = QGridLayout(form_box)

        self.lon_edit = QLineEdit()
        self.lat_edit = QLineEdit()
        self.start_edit = QLineEdit()
        self.end_edit = QLineEdit()
        self.site_edit = QLineEdit("site")

        self.lon_edit.setPlaceholderText("e.g. 150.8981149")
        self.lat_edit.setPlaceholderText("e.g. -23.09416044")
        self.start_edit.setPlaceholderText("YYYY-MM-DD")
        self.end_edit.setPlaceholderText("YYYY-MM-DD")
        self.site_edit.setPlaceholderText("Site name")

        self.freq_combo = QComboBox()
        self.freq_combo.addItems(FREQUENCIES)
        self.freq_combo.setCurrentText("1H")

        self.tz_single_combo = QComboBox()
        self.tz_single_combo.addItems(TIMEZONES)
        self.tz_single_combo.setCurrentText("Australia/Brisbane")

        form_layout.addWidget(QLabel("Longitude"), 0, 0)
        form_layout.addWidget(self.lon_edit, 0, 1)
        form_layout.addWidget(QLabel("Latitude"), 0, 2)
        form_layout.addWidget(self.lat_edit, 0, 3)

        form_layout.addWidget(QLabel("Start Date (YYYY-MM-DD)"), 1, 0)
        form_layout.addWidget(self.start_edit, 1, 1)
        form_layout.addWidget(QLabel("End Date (YYYY-MM-DD)"), 1, 2)
        form_layout.addWidget(self.end_edit, 1, 3)

        form_layout.addWidget(QLabel("Frequency"), 2, 0)
        form_layout.addWidget(self.freq_combo, 2, 1)
        form_layout.addWidget(QLabel("Site Name"), 2, 2)
        form_layout.addWidget(self.site_edit, 2, 3)

        form_layout.addWidget(QLabel("Timezone"), 3, 0)
        form_layout.addWidget(self.tz_single_combo, 3, 1)

        self.pick_map_btn = QPushButton("Pick on Map")
        self.pick_map_btn.setProperty("variant", "secondary")
        self.pick_map_btn.clicked.connect(self.open_map_picker)
        form_layout.addWidget(self.pick_map_btn, 3, 2)

        self.run_btn = QPushButton("Run Analysis")
        self.save_csv_btn = QPushButton("Save to CSV")
        self.save_plot_btn = QPushButton("Save Plot PNG")
        self.save_summary_btn = QPushButton("Save Summary JSON")

        self.run_btn.setProperty("variant", "primary")
        self.save_csv_btn.setProperty("variant", "secondary")
        self.save_plot_btn.setProperty("variant", "secondary")
        self.save_summary_btn.setProperty("variant", "secondary")

        self.run_btn.clicked.connect(self.run_single_analysis)
        self.save_csv_btn.clicked.connect(self.save_single_csv)
        self.save_plot_btn.clicked.connect(self.save_plot_png)
        self.save_summary_btn.clicked.connect(self.save_summary_json)

        action_row = QHBoxLayout()
        action_row.addWidget(self.run_btn)
        action_row.addWidget(self.save_csv_btn)
        action_row.addWidget(self.save_plot_btn)
        action_row.addWidget(self.save_summary_btn)
        action_row.addStretch(1)

        self.single_info_label = QLabel("No analysis yet")
        self.single_info_label.setObjectName("infoPill")

        self.figure = Figure(figsize=(10, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Tide Prediction")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Tide (m)")
        self.ax.grid(alpha=0.25)
        self.canvas = FigureCanvasQTAgg(self.figure)

        layout.addWidget(form_box)
        layout.addLayout(action_row)
        layout.addWidget(self.single_info_label)
        layout.addWidget(self.canvas, stretch=1)

        return panel

    def _build_batch_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        control_box = QGroupBox("Batch Controls")
        control_layout = QGridLayout(control_box)

        self.batch_time_period_radio = QRadioButton("Time Period Predictions")
        self.batch_single_time_radio = QRadioButton("Single Time Point Predictions")
        self.batch_single_time_radio.setChecked(True)

        self.batch_mode_group = QButtonGroup(self)
        self.batch_mode_group.addButton(self.batch_time_period_radio)
        self.batch_mode_group.addButton(self.batch_single_time_radio)
        self.batch_time_period_radio.toggled.connect(self._on_batch_mode_changed)
        self.batch_single_time_radio.toggled.connect(self._on_batch_mode_changed)

        self.tz_batch_combo = QComboBox()
        self.tz_batch_combo.addItems(TIMEZONES)
        self.tz_batch_combo.setCurrentText("Australia/Brisbane")

        self.batch_file_edit = QLineEdit()
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setProperty("variant", "secondary")
        self.browse_btn.clicked.connect(self.select_batch_file)

        self.load_btn = QPushButton("Load CSV")
        self.load_btn.setProperty("variant", "secondary")
        self.load_btn.clicked.connect(self.load_batch_csv)

        self.process_all_btn = QPushButton("Process All Rows")
        self.process_all_btn.setProperty("variant", "primary")
        self.process_all_btn.clicked.connect(self.process_all_batch_rows)

        self.process_selected_btn = QPushButton("Process Selected Rows")
        self.process_selected_btn.setProperty("variant", "primary")
        self.process_selected_btn.clicked.connect(self.process_selected_batch_rows)

        self.batch_row_select_edit = QLineEdit()
        self.batch_row_select_edit.setPlaceholderText("Optional row selection, e.g. 1-20,25,30-40")

        self.batch_progress = QProgressBar()
        self.batch_progress.setMinimum(0)
        self.batch_progress.setMaximum(100)
        self.batch_progress.setValue(0)

        control_layout.addWidget(self.batch_time_period_radio, 0, 0)
        control_layout.addWidget(self.batch_single_time_radio, 0, 1)
        control_layout.addWidget(QLabel("Timezone"), 1, 0)
        control_layout.addWidget(self.tz_batch_combo, 1, 1)
        control_layout.addWidget(QLabel("Input CSV"), 2, 0)
        control_layout.addWidget(self.batch_file_edit, 2, 1)
        control_layout.addWidget(self.browse_btn, 2, 2)
        control_layout.addWidget(self.load_btn, 3, 0)
        control_layout.addWidget(self.process_all_btn, 3, 1)
        control_layout.addWidget(self.process_selected_btn, 3, 2)
        control_layout.addWidget(QLabel("Rows"), 4, 0)
        control_layout.addWidget(self.batch_row_select_edit, 4, 1, 1, 2)
        control_layout.addWidget(self.batch_progress, 5, 0, 1, 3)

        self.batch_preview = QPlainTextEdit()
        self.batch_preview.setObjectName("codePanel")
        self.batch_preview.setReadOnly(True)

        self.batch_log = QPlainTextEdit()
        self.batch_log.setObjectName("codePanel")
        self.batch_log.setReadOnly(True)
        self.batch_log.setPlaceholderText("Batch processing log")

        layout.addWidget(control_box)
        layout.addWidget(QLabel("CSV Preview"))
        layout.addWidget(self.batch_preview, stretch=1)
        layout.addWidget(QLabel("Run Log"))
        layout.addWidget(self.batch_log, stretch=1)

        return panel

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                font-family: "Segoe UI", "Tahoma", sans-serif;
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #eef2f8,
                    stop: 0.45 #f5f8fc,
                    stop: 1 #f2f5fa
                );
            }
            #headerCard {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #15335c,
                    stop: 1 #1d548d
                );
                border: 1px solid #2d5f94;
                border-radius: 12px;
            }
            #headerTitle {
                color: #f7fbff;
                font-size: 22px;
                font-weight: 700;
                letter-spacing: 0.4px;
            }
            #headerSubtitle {
                color: #cfe0f3;
                font-size: 13px;
            }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #d3dcea;
                border-radius: 10px;
                margin-top: 10px;
                padding: 12px;
                background: #fbfdff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #193553;
                font-size: 13px;
            }
            QPushButton {
                background: #2e6ca9;
                color: white;
                border: 1px solid #295f94;
                border-radius: 8px;
                padding: 7px 14px;
                font-weight: 600;
            }
            QPushButton:hover { background: #255e95; }
            QPushButton:pressed { background: #214f7d; }
            QPushButton[variant="primary"] {
                background: #0d6efd;
                border-color: #0a60de;
            }
            QPushButton[variant="primary"]:hover { background: #0b5ed7; }
            QPushButton[variant="primary"]:pressed { background: #0a58ca; }
            QPushButton[variant="secondary"] {
                background: #f4f8ff;
                color: #1d4f80;
                border: 1px solid #b9cde4;
            }
            QPushButton[variant="secondary"]:hover { background: #e9f2ff; }
            QPushButton[variant="secondary"]:pressed { background: #dfeafb; }
            QLineEdit, QComboBox, QPlainTextEdit {
                background: #ffffff;
                border: 1px solid #c9d6e5;
                border-radius: 8px;
                padding: 6px;
                color: #13253a;
            }
            QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus {
                border: 1px solid #4a8ed8;
                background: #fafdff;
            }
            QTabWidget::pane {
                border: 1px solid #cfdaea;
                background: #ffffff;
                border-radius: 10px;
                top: -1px;
            }
            QTabBar::tab {
                background: #eaf1fa;
                color: #335070;
                border: 1px solid #cfdaea;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 8px 16px;
                margin-right: 3px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #0d3d71;
                font-weight: 700;
            }
            QRadioButton {
                color: #1f3855;
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border-radius: 7px;
                border: 1px solid #7f9fc2;
                background: #f7fbff;
            }
            QRadioButton::indicator:checked {
                background: #0d6efd;
                border: 1px solid #0a5dd0;
            }
            QProgressBar {
                border: 1px solid #b9cade;
                border-radius: 7px;
                background: #eef3f9;
                text-align: center;
                color: #22415f;
                font-weight: 600;
            }
            QProgressBar::chunk {
                border-radius: 6px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3f89df, stop:1 #0d6efd);
            }
            #infoPill {
                background: #f2f8ff;
                border: 1px solid #c5d9f1;
                border-radius: 8px;
                padding: 6px 10px;
                color: #224468;
                font-weight: 600;
            }
            #statusBarLabel {
                background: #eef5ff;
                border: 1px solid #c8daee;
                border-radius: 8px;
                color: #214261;
                padding: 6px 10px;
                font-weight: 600;
            }
            #codePanel {
                font-family: Consolas, "Courier New", monospace;
                background: #fcfdff;
                border: 1px solid #cfdbeb;
            }
            """
        )

    def _apply_button_icons(self) -> None:
        style = self.style()
        self.pick_map_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.run_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.save_csv_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.save_plot_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        self.save_summary_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_FileDialogInfoView))
        self.browse_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.load_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.process_all_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.process_selected_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_ArrowForward))
        self.survey_browse_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.survey_draw_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.survey_run_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.survey_save_csv_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))

    def _is_single_time_mode(self) -> bool:
        return self.batch_single_time_radio.isChecked()

    def _on_batch_mode_changed(self, checked: bool) -> None:
        # Only act on the radio that just became active.
        if not checked:
            return

        self.batch_df = None
        self.batch_file_edit.clear()
        self.batch_row_select_edit.clear()
        self.batch_preview.clear()
        self.batch_log.clear()
        self.batch_progress.setValue(0)

        mode = "Single Time Point" if self._is_single_time_mode() else "Time Period"
        self._set_status(f"Batch mode changed to {mode}. Form cleared.")

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _log_batch(self, text: str) -> None:
        self.batch_log.appendPlainText(text)

    def _current_map_center(self) -> tuple[float, float]:
        try:
            lat = float(self.lat_edit.text().strip())
            lon = float(self.lon_edit.text().strip())
            return lat, lon
        except ValueError:
            return -23.0, 150.0

    def _set_coordinate_fields(self, lat: float, lon: float) -> None:
        self.lat_edit.setText(f"{lat:.6f}")
        self.lon_edit.setText(f"{lon:.6f}")

    def _survey_map_center(self) -> tuple[float, float]:
        if self.survey_polygons and self.survey_polygons[0].shape[0] > 0:
            poly = self.survey_polygons[0]
            return float(np.mean(poly[:, 1])), float(np.mean(poly[:, 0]))
        return self._current_map_center()

    def _update_survey_faces(self) -> None:
        self.survey_face_indices = faces_in_polygons(self.model.face_lon, self.model.face_lat, self.survey_polygons)
        n_poly = len(self.survey_polygons)
        n_faces = int(self.survey_face_indices.size)
        self.survey_face_info_label.setText(f"Polygons: {n_poly} | Faces in area: {n_faces}")

    def load_survey_polygon_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Survey Polygon",
            str(INPUT_DIR),
            "Polygon files (*.kml *.shp);;KML (*.kml);;Shapefile (*.shp)",
        )
        if not path:
            return

        try:
            lower = path.lower()
            if lower.endswith(".kml"):
                polygons = _parse_kml_polygons(path)
            elif lower.endswith(".shp"):
                polygons = _parse_shp_polygons(path)
            else:
                raise ValueError("Unsupported polygon file. Use KML or SHP.")

            if not polygons:
                raise ValueError("No valid polygon geometry found in file")

            self.survey_polygon_file_edit.setText(path)
            self.survey_polygons = polygons
            self._update_survey_faces()
            self._set_status(f"Loaded survey polygon file: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Survey Polygon Error", str(exc))

    def draw_survey_polygon(self) -> None:
        lat, lon = self._survey_map_center()
        dialog = PolygonDrawDialog(center_lat=lat, center_lon=lon, parent=self)
        if dialog.exec() == QDialog.Accepted and dialog.polygon is not None:
            self.survey_polygons = [dialog.polygon]
            self.survey_polygon_file_edit.setText("[Drawn on map]")
            self._update_survey_faces()
            self._set_status("Survey polygon drawn on map")

    def clear_survey_polygon(self) -> None:
        self.survey_polygons = []
        self.survey_face_indices = np.array([], dtype=int)
        self.survey_polygon_file_edit.clear()
        self.survey_face_info_label.setText("Survey polygon: none")
        self._set_status("Survey polygon cleared")

    def run_survey_analysis(self) -> None:
        try:
            if self.survey_face_indices.size == 0:
                raise ValueError("No survey polygon loaded/drawn, or no mesh faces fall inside the polygon")

            start = pd.Timestamp(self.survey_start_edit.text().strip())
            end = pd.Timestamp(self.survey_end_edit.text().strip())
            if end < start:
                raise ValueError("End date must be after start date")

            freq = self.survey_freq_combo.currentText().strip()
            tz_name = self.survey_tz_combo.currentText().strip()
            survey_name = self.survey_name_edit.text().strip() or "survey"

            times = pd.date_range(start=start, end=end, freq=freq)
            survey_series = self.model.predict_series_for_faces(
                face_indices=self.survey_face_indices,
                times_local=times,
                timezone_name=tz_name,
            )

            self.survey_results = survey_series
            self.survey_name = survey_name

            self.survey_ax.clear()
            self.survey_ax.plot(survey_series.index, survey_series.values, lw=1.4, color="#0d6efd")
            self.survey_ax.set_title(
                f"{survey_name} | Averaged over {int(self.survey_face_indices.size)} mesh faces"
            )
            self.survey_ax.set_xlabel("Time")
            self.survey_ax.set_ylabel("Tide (m)")
            self.survey_ax.grid(alpha=0.3)
            self.survey_figure.tight_layout()
            self.survey_canvas.draw_idle()

            self._set_status(
                f"Survey run complete: {len(survey_series)} points, {int(self.survey_face_indices.size)} faces"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Survey Analysis Error", str(exc))
            self._set_status("Survey run failed")

    def save_survey_csv(self) -> None:
        if self.survey_results is None:
            QMessageBox.warning(self, "No Data", "Run survey analysis first")
            return

        out_path = OUTPUT_DIR / f"{self.survey_name}_survey_average.csv"
        out_df = self._series_to_export_df(self.survey_results)
        out_df["n_faces_averaged"] = int(self.survey_face_indices.size)
        out_df.to_csv(out_path, index=False)
        self._set_status(f"Saved survey CSV: {out_path}")
        QMessageBox.information(self, "Saved", f"Saved survey CSV to:\n{out_path}")

    @staticmethod
    def _series_to_export_df(tide_series: pd.Series) -> pd.DataFrame:
        dt_text = tide_series.index.strftime("%Y-%m-%d %H:%M:%S")
        return pd.DataFrame({"Date_Time": dt_text, "tide_m": tide_series.values})

    def open_map_picker(self) -> None:
        lat, lon = self._current_map_center()
        dialog = MapPickerDialog(initial_lat=lat, initial_lon=lon, parent=self)
        if dialog.exec() == QDialog.Accepted:
            self._set_coordinate_fields(dialog.selected_lat, dialog.selected_lon)
            self._set_status(
                f"Coordinates selected from map: {dialog.selected_lat:.6f}, {dialog.selected_lon:.6f}"
            )

    def run_single_analysis(self) -> None:
        try:
            lon = float(self.lon_edit.text().strip())
            lat = float(self.lat_edit.text().strip())
            start = pd.Timestamp(self.start_edit.text().strip())
            end = pd.Timestamp(self.end_edit.text().strip())
            freq = self.freq_combo.currentText().strip()
            tz_name = self.tz_single_combo.currentText().strip()
            site = self.site_edit.text().strip() or "site"

            if end < start:
                raise ValueError("End date must be after start date")

            times = pd.date_range(start=start, end=end, freq=freq)
            tide_series, distance_km = self.model.predict_series(lon=lon, lat=lat, times_local=times, timezone_name=tz_name)

            self.single_results = tide_series
            self.single_distance_km = distance_km
            self.single_site_name = site

            self.ax.clear()
            self.ax.plot(tide_series.index, tide_series.values, lw=1.3, color="#0d6efd")
            self.ax.set_title(f"{site} | Distance to nearest grid face: {distance_km:.3f} km")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Tide (m)")
            self.ax.grid(alpha=0.3)
            self.figure.tight_layout()
            self.canvas.draw_idle()

            summary = (
                f"Points: {len(tide_series)} | Min: {tide_series.min():.3f} m | "
                f"Max: {tide_series.max():.3f} m | Mean: {tide_series.mean():.3f} m"
            )
            self.single_info_label.setText(summary)
            self._set_status(f"Single run complete. Distance_km={distance_km:.4f}")
        except Exception as exc:
            QMessageBox.critical(self, "Single Analysis Error", str(exc))
            self._set_status("Single run failed")

    def save_single_csv(self) -> None:
        if self.single_results is None:
            QMessageBox.warning(self, "No Data", "Run analysis first")
            return

        out_path = OUTPUT_DIR / f"{self.single_site_name}.csv"
        out_df = self._series_to_export_df(self.single_results)
        out_df.to_csv(out_path, index=False)
        self._set_status(f"Saved CSV: {out_path}")
        QMessageBox.information(self, "Saved", f"Saved CSV to:\n{out_path}")

    def save_plot_png(self) -> None:
        if self.single_results is None:
            QMessageBox.warning(self, "No Data", "Run analysis first")
            return

        default_name = f"{self.single_site_name}_plot.png"
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot PNG",
            str(OUTPUT_DIR / default_name),
            "PNG (*.png)",
        )
        if not out_path:
            return

        self.figure.savefig(out_path, dpi=160)
        self._set_status(f"Saved plot: {out_path}")

    def save_summary_json(self) -> None:
        if self.single_results is None:
            QMessageBox.warning(self, "No Data", "Run analysis first")
            return

        stats = {
            "site": self.single_site_name,
            "n_points": int(len(self.single_results)),
            "min_tide_m": float(self.single_results.min()),
            "max_tide_m": float(self.single_results.max()),
            "mean_tide_m": float(self.single_results.mean()),
            "distance_km": float(self.single_distance_km) if self.single_distance_km is not None else None,
            "timezone": str(self.single_results.index.tz),
            "start": str(self.single_results.index.min()),
            "end": str(self.single_results.index.max()),
        }

        out_path = OUTPUT_DIR / f"{self.single_site_name}_summary.json"
        out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        self._set_status(f"Saved summary: {out_path}")
        QMessageBox.information(self, "Saved", f"Saved summary JSON to:\n{out_path}")

    def select_batch_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input CSV",
            str(INPUT_DIR),
            "CSV files (*.csv);;All files (*.*)",
        )
        if path:
            self.batch_file_edit.setText(path)

    def load_batch_csv(self) -> None:
        path = self.batch_file_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Missing File", "Select a CSV file first")
            return

        try:
            df = pd.read_csv(path)
            self.batch_df = df
            self.batch_preview.setPlainText(df.head(300).to_string(index=False))
            self._set_status(f"Loaded {len(df)} rows from {path}")
            self._log_batch(f"Loaded CSV: {path} ({len(df)} rows)")
        except Exception as exc:
            QMessageBox.critical(self, "CSV Error", str(exc))

    def process_all_batch_rows(self) -> None:
        if self.batch_df is None:
            self.load_batch_csv()
            if self.batch_df is None:
                return

        indices = list(range(len(self.batch_df)))
        self._process_batch_indices(indices)

    def process_selected_batch_rows(self) -> None:
        if self.batch_df is None:
            self.load_batch_csv()
            if self.batch_df is None:
                return

        try:
            indices = parse_row_selection(self.batch_row_select_edit.text(), len(self.batch_df))
        except Exception as exc:
            QMessageBox.warning(self, "Row Selection Error", str(exc))
            return

        self._process_batch_indices(indices)

    def _process_batch_indices(self, indices: list[int]) -> None:
        assert self.batch_df is not None
        tz_name = self.tz_batch_combo.currentText().strip()
        mode_single = self._is_single_time_mode()

        try:
            df = self.batch_df.iloc[indices].copy()
            total = len(df)
            if total == 0:
                raise ValueError("No rows selected")

            self.batch_progress.setValue(0)
            self._log_batch(f"Starting batch run: {total} rows")

            if mode_single:
                required = {"Site", "Latitude", "Longitude", "Date_Time"}
                if not required.issubset(df.columns):
                    raise ValueError(f"Single time mode requires columns: {sorted(required)}")

                out_rows = []
                for i, (_, row) in enumerate(df.iterrows(), start=1):
                    site = str(row["Site"])
                    lat = float(row["Latitude"])
                    lon = float(row["Longitude"])
                    dt = parse_datetime_flexible(str(row["Date_Time"]))

                    ts = pd.DatetimeIndex([dt])
                    tide_series, dist_km = self.model.predict_series(lon=lon, lat=lat, times_local=ts, timezone_name=tz_name)

                    out_rows.append(
                        {
                            "Site": site,
                            "Latitude": lat,
                            "Longitude": lon,
                            "Date_Time": str(row["Date_Time"]),
                            "Timezone": tz_name,
                            "Tide_m": float(np.round(tide_series.iloc[0], 4)),
                            "Distance_km": float(np.round(dist_km, 4)),
                        }
                    )

                    self.batch_progress.setValue(int(i * 100 / total))
                    if i % max(1, total // 10) == 0 or i == total:
                        self._log_batch(f"Processed {i}/{total} rows")
                    QApplication.processEvents()

                out_df = pd.DataFrame(out_rows)
                out_path = OUTPUT_DIR / "tide_predictions_single_time.csv"
                out_df.to_csv(out_path, index=False)
                self._set_status(f"Batch complete: {len(out_df)} rows -> {out_path}")
                self._log_batch(f"Wrote output: {out_path}")
                QMessageBox.information(self, "Completed", f"Saved:\n{out_path}")
                return

            required = {"Site", "Longitude", "Latitude", "start", "stop", "interval"}
            if not required.issubset(df.columns):
                raise ValueError(f"Time period mode requires columns: {sorted(required)}")

            outputs_written = 0
            for i, (_, row) in enumerate(df.iterrows(), start=1):
                site = str(row["Site"]).strip() or "site"
                safe_site = re.sub(r"[^A-Za-z0-9_.-]+", "_", site)
                lon = float(row["Longitude"])
                lat = float(row["Latitude"])
                start = pd.Timestamp(str(row["start"]))
                stop = pd.Timestamp(str(row["stop"]))
                interval = str(row["interval"]).strip()

                if stop < start:
                    self._log_batch(f"Skipped {site}: stop before start")
                    continue

                times = pd.date_range(start=start, end=stop, freq=interval)
                tide_series, _ = self.model.predict_series(lon=lon, lat=lat, times_local=times, timezone_name=tz_name)

                out_path = OUTPUT_DIR / f"{safe_site}.csv"
                out_df = self._series_to_export_df(tide_series)
                out_df.to_csv(out_path, index=False)
                outputs_written += 1

                self.batch_progress.setValue(int(i * 100 / total))
                if i % max(1, total // 10) == 0 or i == total:
                    self._log_batch(f"Processed {i}/{total} rows")
                QApplication.processEvents()

            self._set_status(f"Batch complete: wrote {outputs_written} files")
            self._log_batch(f"Wrote {outputs_written} files to {OUTPUT_DIR}")
            QMessageBox.information(self, "Completed", f"Wrote {outputs_written} files to:\n{OUTPUT_DIR}")
        except Exception as exc:
            QMessageBox.critical(self, "Batch Error", str(exc))
            self._set_status("Batch run failed")
            self._log_batch(f"Error: {exc}")


def main() -> None:
    app = QApplication(sys.argv)
    window = TidePredictionWindow()
    window.show()
    sys.exit(app.exec())


# Backwards compatibility for previous class name.
TideAnalysisWindow = TidePredictionWindow


if __name__ == "__main__":
    main()
