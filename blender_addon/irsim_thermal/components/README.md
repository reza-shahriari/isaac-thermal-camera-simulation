# Hidden-part placeholders

One FBX per component a downloaded model usually lacks: what sits under the skin and makes or
stores heat, for drones, aircraft, helicopters, cars, trucks, motorcycles, trains, ships, and
people inside vehicles and buildings. Each is a **placeholder**, a box, a cylinder or a cone at a
typical real size, in metres, centred on its origin, with no material. Replace any of them with a
real model later: keep the file name, metres and the centre, and anything placed from it can be
swapped for the real one.

In Blender, **Hidden parts ▸ Add from library…** places one in the middle of the selected part,
at its real size, and marks it as a hidden part; you then give it a material, a mass and its heat.
Where irsim's component library (`configs/components/`) has the same thing, the catalog names it
("Takes irsim's …" below) and the part takes that component's cited mass, heat capacity and heat
unless you type your own.
The tutorial step
[*Add the parts you cannot see*](../../../docs/tutorials/blender-addon/07-hidden-parts.md) explains
those numbers.

`catalog.json` lists the components: the menu group, the name, what each one is, and the shape
and size of its placeholder. A new one is one line there, then

```bash
blender -b --factory-startup --python blender_addon/tools/make_placeholders.py
```

which writes the FBX files that are missing (never over an existing one unless given `-- --force`,
since that may be your own model), imports each back to check it returns at its catalog size with
no rotation, and rewrites the table below.

The sizes are **estimates** for a typical example of each, and so are placeholders too. X is each
component's length unless the table names another axis.

<!-- table: written by tools/make_placeholders.py -->

**Drone**

| file | shape | size (m) | stands for |
|---|---|---|---|
| `drone_battery.fbx` | box | 0.14 × 0.07 × 0.04 | Flight battery: A multirotor's flight battery (Phantom 4 class). Takes irsim's `lipo_pack`. |
| `drone_battery_small.fbx` | box | 0.075 × 0.035 × 0.035 | Small flight battery: A racing or FPV drone's 4S pack. |
| `drone_battery_large.fbx` | box | 0.2 × 0.09 × 0.06 | Heavy-lift battery: A large 6S pack for a heavy-lift or agricultural drone. |
| `drone_esc.fbx` | box | 0.04 × 0.02 × 0.006 | Speed controller: One electronic speed controller, in an arm. Takes irsim's `esc`. |
| `drone_esc_4in1.fbx` | box | 0.036 × 0.036 × 0.008 | Four-in-one speed controller: Four speed controllers on one board, in the body. |
| `drone_motor_small.fbx` | cylinder | 0.018 along Z, ⌀ 0.028 | Small motor: A racing drone's brushless motor; shaft along Z. |
| `drone_motor.fbx` | cylinder | 0.024 along Z, ⌀ 0.028 | Motor: A camera drone's brushless motor (Phantom 4 class); shaft along Z. Takes irsim's `brushless_motor`. |
| `drone_motor_large.fbx` | cylinder | 0.03 along Z, ⌀ 0.07 | Heavy-lift motor: A heavy-lift drone's brushless motor; shaft along Z. |
| `drone_flight_controller.fbx` | box | 0.05 × 0.05 × 0.015 | Flight controller: The flight-controller board. Takes irsim's `flight_controller`. |
| `drone_video_transmitter.fbx` | box | 0.06 × 0.04 × 0.012 | Video transmitter: The video downlink transmitter. |
| `drone_companion_computer.fbx` | box | 0.1 × 0.087 × 0.03 | Onboard computer: A companion computer with its heat sink (Jetson class). |
| `drone_power_board.fbx` | box | 0.05 × 0.05 × 0.005 | Power distribution board: The board that carries the battery current to the arms. |
| `drone_radio_receiver.fbx` | box | 0.03 × 0.02 × 0.008 | Radio receiver: The remote-control receiver. |
| `drone_gnss_module.fbx` | box | 0.05 × 0.05 × 0.015 | GPS module: The satellite-navigation receiver and compass. |

**Fixed-wing drone**

| file | shape | size (m) | stands for |
|---|---|---|---|
| `uav_piston_engine.fbx` | box | 0.2 × 0.15 × 0.15 | Small piston engine: A 50 cc two-stroke engine of a fixed-wing drone. Takes irsim's `piston_engine`. |
| `uav_electric_motor.fbx` | cylinder | 0.06 along X, ⌀ 0.05 | Electric motor: A fixed-wing drone's pusher or tractor motor; shaft along X. |
| `uav_battery.fbx` | box | 0.2 × 0.1 × 0.05 | Battery: A fixed-wing drone's flight battery. |
| `uav_fuel_tank.fbx` | box | 0.25 × 0.15 × 0.1 | Fuel tank: Fuel for the piston engine: thermal mass. |

**Aircraft**

| file | shape | size (m) | stands for |
|---|---|---|---|
| `aircraft_jet_engine.fbx` | cone | 2.5 along X, ⌀ 1 → 0.6 | Small jet engine: A business jet's turbofan; inlet at -X, nozzle at +X. |
| `aircraft_turbofan_airliner.fbx` | cone | 3.5 along X, ⌀ 1.9 → 1 | Airliner jet engine: A narrow-body airliner's turbofan; inlet at -X, nozzle at +X. |
| `aircraft_turbofan_widebody.fbx` | cone | 5.5 along X, ⌀ 3.3 → 1.6 | Wide-body jet engine: A wide-body airliner's large turbofan; inlet at -X, nozzle at +X. |
| `aircraft_turboprop_engine.fbx` | cylinder | 1.7 along X, ⌀ 0.5 | Turboprop engine: A turboprop's gas generator and gearbox; propeller at -X. |
| `aircraft_piston_engine.fbx` | box | 0.75 × 0.85 × 0.6 | Light-aircraft engine: A four-cylinder flat engine of a light aircraft. |
| `aircraft_exhaust_duct.fbx` | cylinder | 3 along X, ⌀ 0.8 | Engine exhaust duct: The hot duct behind a turbine inside the fuselage; flow along +X. |
| `aircraft_apu.fbx` | box | 1 × 0.6 × 0.6 | Auxiliary power unit: An aircraft's auxiliary power unit, in the tail. |
| `aircraft_fuel_tank.fbx` | box | 3 × 1.5 × 0.3 | Wing fuel tank: Fuel: thermal mass, cold-soaked at altitude. |
| `aircraft_air_conditioning_pack.fbx` | box | 1.2 × 0.8 × 0.6 | Air-conditioning pack: The cabin air-conditioning unit, under the wing box. |
| `aircraft_avionics_rack.fbx` | box | 0.6 × 0.5 × 0.6 | Avionics rack: The electronics bay under the cockpit. |
| `aircraft_battery.fbx` | box | 0.3 × 0.25 × 0.25 | Aircraft battery: The main battery. |
| `aircraft_hydraulic_pump.fbx` | box | 0.3 × 0.2 × 0.2 | Hydraulic pump: An electric hydraulic pump. |
| `aircraft_wheel_brake.fbx` | cylinder | 0.2 along Y, ⌀ 0.5 | Wheel brake: A main-gear wheel brake, hot after landing; axle along Y. |

**Helicopter**

| file | shape | size (m) | stands for |
|---|---|---|---|
| `heli_turboshaft_engine.fbx` | cylinder | 1.2 along X, ⌀ 0.6 | Turboshaft engine: A helicopter's gas turbine; exhaust at +X. |
| `heli_piston_engine.fbx` | box | 0.8 × 0.8 × 0.6 | Piston engine: A light helicopter's piston engine. |
| `heli_main_gearbox.fbx` | box | 1 × 0.8 × 0.7 | Main rotor gearbox: Under the main rotor. |
| `heli_tail_gearbox.fbx` | box | 0.3 × 0.25 × 0.25 | Tail rotor gearbox: At the top of the tail fin. |
| `heli_exhaust.fbx` | cylinder | 1 along X, ⌀ 0.4 | Exhaust pipe: The turbine's exhaust; flow along +X. |
| `heli_oil_cooler.fbx` | box | 0.5 × 0.4 × 0.2 | Oil cooler: The gearbox and engine oil cooler. |

**Car**

| file | shape | size (m) | stands for |
|---|---|---|---|
| `car_small_engine.fbx` | box | 0.5 × 0.5 × 0.55 | Small engine: A three-cylinder engine of a small car. |
| `car_piston_engine.fbx` | box | 0.6 × 0.55 × 0.6 | Engine: A four-cylinder engine block. |
| `car_v8_engine.fbx` | box | 0.75 × 0.75 × 0.7 | V8 engine: A large V8 engine. |
| `car_gearbox.fbx` | box | 0.5 × 0.35 × 0.35 | Gearbox: A gearbox. |
| `car_differential.fbx` | box | 0.35 × 0.3 × 0.3 | Differential: The rear axle's differential. Takes irsim's `differential`. |
| `car_turbocharger.fbx` | box | 0.25 × 0.25 × 0.2 | Turbocharger: Driven by the exhaust, the hottest part of the engine bay. |
| `car_exhaust_manifold.fbx` | box | 0.45 × 0.15 × 0.15 | Exhaust manifold: Where the cylinders' exhaust gathers. |
| `car_catalytic_converter.fbx` | cone | 0.35 along X, ⌀ 0.14 → 0.1 | Catalytic converter: Flow along +X. |
| `car_exhaust_pipe.fbx` | cylinder | 3 along X, ⌀ 0.055 | Exhaust pipe: The pipe under the floor; flow along +X. |
| `car_exhaust_silencer.fbx` | box | 0.5 × 0.25 × 0.15 | Exhaust silencer: The rear silencer. |
| `car_radiator.fbx` | box | 0.04 × 0.65 × 0.45 | Radiator: Behind the grille, facing X. |
| `car_intercooler.fbx` | box | 0.06 × 0.6 × 0.2 | Intercooler: In front of the radiator, facing X. |
| `car_brake_disc.fbx` | cylinder | 0.028 along Y, ⌀ 0.32 | Brake disc: A front brake disc and calliper, hot after braking; axle along Y. Takes irsim's `brake_disc`. |
| `car_12v_battery.fbx` | box | 0.28 × 0.175 × 0.19 | 12 V battery: The starter battery. |
| `car_fuel_tank.fbx` | box | 0.8 × 0.6 × 0.25 | Fuel tank: Under the rear seat: thermal mass. |
| `car_hvac_unit.fbx` | box | 0.3 × 0.6 × 0.3 | Heater and air conditioning: The heater core and evaporator behind the dashboard. |
| `car_ecu.fbx` | box | 0.2 × 0.15 × 0.04 | Engine computer: The engine control unit. |

**Electric car**

| file | shape | size (m) | stands for |
|---|---|---|---|
| `car_ev_battery_pack.fbx` | box | 1.8 × 1.3 × 0.12 | Battery pack: An electric car's floor pack. Takes irsim's `ev_pack`. |
| `car_ev_drive_unit.fbx` | box | 0.6 × 0.45 × 0.4 | Drive unit: Motor, inverter and reduction gear in one housing, between the wheels. |
| `car_ev_motor.fbx` | cylinder | 0.4 along Y, ⌀ 0.25 | Motor: A drive motor on its own; shaft along Y. Takes irsim's `ev_drive_motor`. |
| `car_ev_inverter.fbx` | box | 0.35 × 0.25 × 0.1 | Inverter: The motor's power electronics. |
| `car_ev_charger.fbx` | box | 0.35 × 0.25 × 0.1 | On-board charger: Warm while the car charges. |

**Truck and bus**

| file | shape | size (m) | stands for |
|---|---|---|---|
| `truck_diesel_engine.fbx` | box | 1.4 × 0.9 × 1.1 | Diesel engine: A heavy truck's or bus's six-cylinder diesel. |
| `truck_gearbox.fbx` | box | 1 × 0.5 × 0.5 | Gearbox: A heavy truck's gearbox. |
| `truck_exhaust_aftertreatment.fbx` | box | 1 × 0.6 × 0.6 | Exhaust cleaning box: The particle filter and catalyst box beside the chassis. |
| `truck_exhaust_stack.fbx` | cylinder | 1.5 along Z, ⌀ 0.13 | Exhaust stack: An upright exhaust pipe; flow along +Z. |
| `truck_radiator.fbx` | box | 0.08 × 1 × 1 | Radiator: Behind the grille, facing X. |
| `truck_fuel_tank.fbx` | box | 1.2 × 0.65 × 0.65 | Fuel tank: A side-mounted diesel tank: thermal mass. |
| `truck_refrigeration_unit.fbx` | box | 0.45 × 1.8 × 1.7 | Trailer refrigeration unit: On the trailer's front wall, facing X; its condenser is warm. |
| `bus_roof_battery.fbx` | box | 2 × 1 × 0.35 | Electric bus roof battery: One of an electric bus's roof battery packs. |

**Motorcycle**

| file | shape | size (m) | stands for |
|---|---|---|---|
| `motorbike_engine.fbx` | box | 0.45 × 0.45 × 0.5 | Engine: A motorcycle's engine and gearbox. |
| `motorbike_exhaust_silencer.fbx` | cylinder | 0.45 along X, ⌀ 0.1 | Exhaust silencer: Flow along +X. |
| `motorbike_radiator.fbx` | box | 0.05 × 0.35 × 0.3 | Radiator: Facing X. |
| `ebike_battery.fbx` | box | 0.4 × 0.1 × 0.1 | E-bike or scooter battery: In the frame or under the deck. |
| `ebike_hub_motor.fbx` | cylinder | 0.05 along Y, ⌀ 0.2 | Hub motor: In a wheel's hub; axle along Y. |

**Train**

| file | shape | size (m) | stands for |
|---|---|---|---|
| `train_diesel_engine.fbx` | box | 5 × 1.7 × 2.5 | Locomotive diesel engine: A diesel locomotive's main engine. |
| `train_traction_motor.fbx` | cylinder | 1 along Y, ⌀ 0.8 | Traction motor: One axle's motor; axle along Y. |
| `train_transformer.fbx` | box | 2.5 × 1.5 × 0.8 | Main transformer: An electric train's transformer, under the floor. |

**Ship and boat**

| file | shape | size (m) | stands for |
|---|---|---|---|
| `ship_marine_engine.fbx` | box | 4 × 1.5 × 2 | Ship engine: A main engine. |
| `ship_generator.fbx` | box | 2 × 1 × 1.2 | Ship generator: A diesel generator set. |
| `ship_gas_turbine.fbx` | cylinder | 6.5 along X, ⌀ 2 | Ship gas turbine: A marine gas turbine; exhaust at +X. |
| `ship_gearbox.fbx` | box | 2.5 × 2 × 2 | Ship gearbox: Between the main engine and the shaft. |
| `ship_electric_motor.fbx` | cylinder | 3 along X, ⌀ 2.5 | Ship propulsion motor: An electric propulsion motor; shaft along X. |
| `ship_boiler.fbx` | box | 3 × 2.5 × 3 | Boiler: An auxiliary or exhaust-gas boiler. |
| `ship_funnel_uptake.fbx` | cylinder | 3 along Z, ⌀ 1.2 | Funnel exhaust uptake: The exhaust pipe inside the funnel; flow along +Z. |
| `ship_fuel_tank.fbx` | box | 5 × 4 × 2 | Ship fuel tank: A heated fuel-oil tank. |
| `boat_inboard_engine.fbx` | box | 1 × 0.7 × 0.7 | Boat inboard engine: A motor boat's or yacht's engine. |
| `boat_outboard_powerhead.fbx` | box | 0.5 × 0.4 × 0.5 | Outboard engine: The engine under an outboard's cowling. |
| `boat_jetski_engine.fbx` | box | 0.6 × 0.45 × 0.45 | Jet-ski engine: A personal watercraft's engine. |
| `boat_battery_bank.fbx` | box | 0.6 × 0.4 × 0.3 | Boat batteries: A boat's house battery bank. |

**People and animals**

| file | shape | size (m) | stands for |
|---|---|---|---|
| `person_seated.fbx` | box | 0.5 × 0.45 × 0.9 | Seated person: A driver, pilot or passenger inside a vehicle, facing +X. |
| `person_standing.fbx` | box | 0.3 × 0.5 × 1.75 | Standing person: Someone inside a cabin or a building, facing +X. |
| `dog.fbx` | box | 0.8 × 0.25 × 0.6 | Dog: A medium dog, facing +X. |

**Buildings and equipment**

| file | shape | size (m) | stands for |
|---|---|---|---|
| `equipment_generator_set.fbx` | box | 2 × 1 × 1.3 | Diesel generator: A standby generator inside its enclosure. |
| `equipment_air_compressor.fbx` | cylinder | 0.4 along Z, ⌀ 0.25 | Air conditioner compressor: The compressor inside an outdoor unit; upright. |
| `equipment_transformer.fbx` | box | 1.2 × 1 × 1.5 | Power transformer: A distribution transformer inside a kiosk. |
| `equipment_server_rack.fbx` | box | 1 × 0.6 × 2 | Server rack: A full rack of computers. |
| `equipment_solar_inverter.fbx` | box | 0.2 × 0.5 × 0.6 | Solar inverter: A wall-mounted inverter. |
| `equipment_battery_storage.fbx` | box | 6 × 2.4 × 2.6 | Battery storage container: A grid battery inside a shipping container. |
| `equipment_water_heater.fbx` | cylinder | 1.5 along Z, ⌀ 0.6 | Water heater: A hot-water tank; upright. |

<!-- end of table -->
