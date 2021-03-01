from enum import Enum

TIME_FORMAT = '%Y-%m-%d %H:%M:%S%z'
HOURS_IN_DAY = 24
SECONDS_IN_HOUR = 3600
SECOND_IN_DAY = 86400
MINUTES_IN_HOUR = 60
MINUTES_IN_DAR = 1440


class XmlElements:
    bids_offers = "bids_offers"
    market_submit = "market_submit"
    sem_demand_offer = "sem_demand_offer"
    identifier = "identifier"
    cod_complex_detail = "cod_complex_detail"
    inc_curve_detail = "inc_curve_detail"
    dec_curve_detail = "dec_curve_detail"
    point = "point"
    shutdown_cost = "shutdown_cost"
    demand_detail = "demand_detail"
    sem_pn_submit = "sem_pn_submit"
    pn_interval = "pn_interval"


class Weekday(Enum):
    monday = 0
    tuesday = 1
    wednesday = 2
    thursday = 3
    friday = 4
    saturday = 5
    sunday = 6


class DataFrameKeys:
    time = 'time_aggregated'
    site = 'site'
    market = 'market'
    op_cert = 'op_cert'
    dsu_id = 'dsu_id'
    op_cert_const_avail = 'op_cert_constrained_availability'
    esb_const_avail = 'esb_constrained_availability'
    total_available = 'unconstrained_availability'


