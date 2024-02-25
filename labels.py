casualty_class = {1: 'Driver or rider', 2: 'Passenger', 3: 'Pedestrian'}
sex_of_casualty = { 1.: 'male', 2.: 'female', 9.: 'unknown'}
casualty_severity = {1: 'Fatal' ,  2: 'serious', 3: 'slight'}
pedestrian_location = {
    0: 'Not a Pedestrian',
    1: 'Crossing on pedestrian crossing facility',
    2: 'Crossing in zig-zag approach lines',
    3: 'Crossing in zig-zag exit lines',
    4: 'Crossing elsewhere within 50m. of pedestrian crossing',
    5: 'In carriageway , crossing elsewhere',
    6: 'On footway or verge',
    7: 'On refuge, central island or central reservation',
    8: 'In centre of carriageway - not on refuge, island or central reservation',
    9: 'In carriageway, not crossing',
    10: 'Unknown or other',
}
pedestrian_movement = {
    0: 'Not a Pedestrian',
    1: 'Crossing from drivers nearside',
    2: 'Crossing from nearside - masked by parked or stationary vehicle',
    3: 'Crossing from drivers offside',
    4: 'Crossing from offside - masked by  parked or stationary vehicle',
    5: 'In carriageway, stationary - not crossing  (standing or playing)',
    6: 'In carriageway, stationary - not crossing  (standing or playing) - masked by parked or stationary vehicle',
    7: 'Walking along in carriageway, facing traffic',
    8: 'Walking along in carriageway, back to traffic',
    9: 'Unknown or other',
}
car_passenger = {
    0.: 'Not car passenger',
    1.: 'Front seat passenger',
    2.: 'Rear seat passenger',
    9.: 'unknown (self reported)',
}
bus_or_coach_passenger = {
    0.: 'Not a bus or coach passenger',
    1.: 'Front seat passenger',
    2.: 'Alighting',
    3.: 'Standing passenger',
    4.: 'Seated passenger',
    9.: 'unknown (self reported)',
}
pedestrian_road_maintenance_worker = {
    0.: 'No / Not applicable',
    1.: 'Yes',
    2.: 'Not Known',
}
casualty_type = {
    0.: 'Pedestrian',
    1.: 'Cyclist',
    2.: 'Motorcycle 50cc and under rider or passenger',
    3.: 'Motorcycle 125cc and under rider or passenger',
    4.: 'Motorcycle over 125cc and up to 500cc rider or  passenger',
    5.: 'Motorcycle over 500cc rider or passenger',
    8.: 'Taxi/Private hire car occupant',
    9.: 'Car occupant',
    10.: 'Minibus (8 - 16 passenger seats) occupant',
    11.: 'Bus or coach occupant (17 or more pass seats)',
    16.: 'Horse rider',
    17.: 'Agricultural vehicle occupant',
    18.: 'Tram occupant',
    19.: 'Van / Goods vehicle (3.5 tonnes mgw or under) occupant',
    20.: 'Goods vehicle (over 3.5t. and under 7.5t.) occupant',
    21.: 'Goods vehicle (7.5 tonnes mgw and over) occupant',
    22.: 'Mobility scooter rider',
    23.: 'Electric motorcycle rider or passenger',
    90.: 'Other vehicle occupant',
    97.: 'Motorcycle - unknown cc rider or passenger',
    98.: 'Goods vehicle (unknown weight) occupant',
}
casualty_home_area_type = {1.: 'Urban area',  2.: 'Small town',  3.: 'Rural', }
age_band_of_casualty = {
    1: '0 - 5', 2: '6 - 10', 3: '11 - 15', 4: '16 - 20',
    5: '21 - 25', 6: '26 - 35', 7: '36 - 45', 8: '46 - 55',
    9: '56 - 65', 10: '66 - 75', 11: 'Over 75',
}