"""Thermal: weather, convection, solar loading, longwave down, temperature solvers.

docs/physics-model.md §6
"""

from irsim.thermal.aerial import (
    AERIAL_HEAT_SOURCES,
    BATTERY,
    ESC,
    MOTOR,
    HeatSource,
    airframe_solver,
    heat_source_solver,
    node_temperature,
    prescribed_from_schedule,
    refine_nodes,
    throttle_profile,
)
from irsim.thermal.conduction import ConductionOperator, explicit_bound_s
from irsim.thermal.convection import (
    DEFAULT_CONVECTION,
    ConvectionParams,
    convection_coefficient,
    free_forced_crossover_k,
    relative_air_speed,
)
from irsim.thermal.longwave import (
    clear_sky_emissivity,
    longwave_down,
    longwave_down_from_sample,
    sky_emissivity,
)
from irsim.thermal.network import (
    FixedNode,
    ImposedHeat,
    Link,
    LinkNode,
    Node,
    RadiationLink,
    ThermalNetwork,
)
from irsim.thermal.shadow import ShadowRectangle, cell_shadow, patch_solar_loading
from irsim.thermal.solar import (
    SunPosition,
    absorbed_solar,
    julian_day,
    solar_loading,
    solar_noon_utc,
    sun_direction,
    sun_position,
    sun_position_utc,
)
from irsim.thermal.solvers import (
    SOLVER_TYPES,
    NewtonCoolingSolver,
    PrescribedSolver,
    SolverState,
    TemperatureSolver,
)
from irsim.thermal.spatial_sources import (
    RadiantRectangle,
    corner_view_factor,
    occluded_longwave_flux,
    patch_view_factors,
    view_factor_to_parallel_rectangle,
)
from irsim.thermal.surface_field import PlanarPatch, PlanarThermalField
from irsim.thermal.weather import WEATHER_FIELDS, WeatherSample, WeatherSeries, seconds_since
from irsim.thermal.weather_io import load_weather_csv, synthetic_clear_day, write_weather_csv

__all__ = [
    "AERIAL_HEAT_SOURCES",
    "BATTERY",
    "ESC",
    "MOTOR",
    "HeatSource",
    "airframe_solver",
    "heat_source_solver",
    "node_temperature",
    "prescribed_from_schedule",
    "refine_nodes",
    "throttle_profile",
    "PlanarPatch",
    "RadiantRectangle",
    "corner_view_factor",
    "occluded_longwave_flux",
    "patch_view_factors",
    "view_factor_to_parallel_rectangle",
    "PlanarThermalField",
    "ConductionOperator",
    "FixedNode",
    "ImposedHeat",
    "Link",
    "LinkNode",
    "Node",
    "RadiationLink",
    "ThermalNetwork",
    "explicit_bound_s",
    "SOLVER_TYPES",
    "NewtonCoolingSolver",
    "PrescribedSolver",
    "SolverState",
    "TemperatureSolver",
    "clear_sky_emissivity",
    "longwave_down",
    "longwave_down_from_sample",
    "sky_emissivity",
    "SunPosition",
    "absorbed_solar",
    "julian_day",
    "solar_loading",
    "ShadowRectangle",
    "cell_shadow",
    "patch_solar_loading",
    "solar_noon_utc",
    "sun_direction",
    "sun_position",
    "sun_position_utc",
    "DEFAULT_CONVECTION",
    "ConvectionParams",
    "convection_coefficient",
    "free_forced_crossover_k",
    "relative_air_speed",
    "WEATHER_FIELDS",
    "WeatherSample",
    "WeatherSeries",
    "seconds_since",
    "load_weather_csv",
    "synthetic_clear_day",
    "write_weather_csv",
]
