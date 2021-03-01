import datetime
import pickle
import pandas
import numpy
import json
import h5py
import sqlalchemy
import os
import glob
import logging
import re
import platform

from cachetools import cached, TTLCache
from scipy import signal

########   GLOBAL AND SYSTEM VARIABELS    #########
POSTGRES_ADDRESS = 'reveal-aws-redshift-cluster-snapshot.cdtsot1ovrch.us-west-2.redshift.amazonaws.com'
POSTGRES_PORT = '5439'
POSTGRES_USERNAME = 'software'
POSTGRES_PASSWORD = 'iSense1755'
POSTGRES_DBNAME = 'reveal'

MIN_TIME_LEN = 40  # time where decisions are made
OPERATING_SYSTEM = platform.system()
DATA_ROOT_DIR = os.getenv('DATA')
if 'windows' in OPERATING_SYSTEM.lower():
    HOME_PATH = os.path.join(os.getenv('HOMEDRIVE'), os.getenv('HOMEPATH'))
else:
    HOME_PATH = os.getenv('HOME')
if DATA_ROOT_DIR is None and 'windows' in OPERATING_SYSTEM:
    DATA_ROOT_DIR = "C:\\Python\pyCific\data"
else:
    DATA_ROOT_DIR = os.path.join('.', 'data')
SPOTS_TO_USE = [32, 33, 51, 91, 108, 70]  # 32, 33, 70, 51, 108, 91 group reds, greens, blues
SPOTS_TO_IGNORE = [-1, -2, '-1', '-2', '32.25.0']
CHEM_TO_IGNORE = {32: [25]}
COLORS = ['ri', 'gi', 'bi']
PANEL_CONFIG_FOLDER = os.getenv('PANEL_CONFIG_FOLDER')
if PANEL_CONFIG_FOLDER is None:
    PANEL_CONFIG_FOLDER = os.path.join(DATA_ROOT_DIR, 'config', 'PlateLayout')
DRUGS_TO_IGNORE = ['Water', 'None', 'LOC']
PANELS_TO_USE = ['NM43', 'PM34']

LTE = "≤"
GTE = "≥"

###########################################


def get_filter(df: pandas.DataFrame, end_point: str, value: 'str'):
    return df[end_point].str.contains(value)


def load_heat_map_from_pickle(species: str, drug: str, start=0, end=MIN_TIME_LEN):
    filename = os.path.join(DATA_ROOT_DIR, 'heat_maps', '{}_{}_{}_{}.pkl'.format(
        species.replace('.', '').replace(' ', ''), drug, start, end))
    return pickle.load(open(filename, 'rb'))


def save_heat_map_to_pickle(secies: str, drug: str, well:str):
    pass


def store_heat_maps(species, drug, panel, run_ids: list = None):
    heat_maps = generate_heat_map(species, drug, panel, save_heat_map=True, run_id_list=run_ids)
    return heat_maps


def _extract_concentraion(input_):
    if isinstance(input_, float):
        return input_
    elif isinstance(input_, int):
        return float(input_)
    else:
        split_input = input_.split('_')
    return float(split_input[0])


def load_panel_config(panel: str):
    raw_panel = pandas.read_csv(os.path.join(PANEL_CONFIG_FOLDER, '{}.csv'.format(panel)))
    wells = []
    concentration = []
    drug = []
    for idx in raw_panel.index:
        if not(raw_panel.loc[idx, EndPointsPanel.abx] in DRUGS_TO_IGNORE):
            wells.append(raw_panel.loc[idx, EndPointsPanel.well])
            drug.append(raw_panel.loc[idx, EndPointsPanel.abx])
            concentration.append(_extract_concentraion(raw_panel.loc[idx, EndPointsPanel.abx_concentration]))
    panel_config = pandas.DataFrame({'drug': drug, 'concentration': concentration}, index=wells)
    return panel_config


def wells_from_panel_for_drug(panel_config: pandas.DataFrame, drug):
    wells = [well for well in panel_config.index[panel_config['drug'] == drug]]
    return wells


class EndPointsPanel:
    well = 'Well'
    analyte = 'Analyte'
    strain = 'Strain'
    abx = 'Abx'
    abx_concentration = 'Abx_concentration'
    replicate = 'Replicate'


class EndPointsMicData:
    uniqueid = 'uniqueid'
    testdata = 'testdate'
    species = 'species'
    abx = 'abx'
    mic_reference = 'mic_reference'
    mic_reveal = 'mic_reveal'


class EndPointsMetadata:
    ignore = 'ignore'
    uniqueid = 'uniqueid'
    testdate = 'testdate'
    study = 'study'
    species = 'species'
    librarybuild = 'librarybuild'
    source = 'bacterial_source'


class EndPointsTimeSeries:
    uniqueid = 'uniqueid'
    testdate = 'testdate'
    abx = 'abx'
    species = 'species'
    well = 'well'
    abx_concentraion = 'abx_concetraion'
    spotid = 'spotid'
    ignore = 'ignore'
    librarybuild = 'librarybuild'
    ref_time = 'timestampdt-ref'
    red = 'ri'
    blue = 'bi'
    green = 'gi'
    platemap = 'platemap'


@cached(cache=TTLCache(maxsize=1024, ttl=86400))
def mic_data_from_db(species: str = None, uniqueid: str = None, drug: str = None, testdate='20200101'):
    """
    retrieves mic reference data from database
    :param species:
    :param uniqueid:
    :param drug:
    :param testdate:
    :return: pandas.DataFrame
    """
    logger = logging.getLogger('{}.mic_data_from_db'.format(__name__))
    if species is None and uniqueid is None and drug is None:
        raise ValueError('must provide species or uniqueid')

    postgres_str = ('postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'.format(
        username=POSTGRES_USERNAME,
        password=POSTGRES_PASSWORD,
        ipaddress=POSTGRES_ADDRESS,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DBNAME))
    engine = sqlalchemy.create_engine(postgres_str)
    sql_query = 'SELECT * FROM experiment_mic where testdate > \'{}\' AND '.format(testdate)
    if not (species is None):
        sql_query += 'species = \'{}\''.format(species)
    else:
        sql_query += 'uniqueid = \'{}\''.format(uniqueid)
    if drug is None:
        sql_query += ';'
    else:
        sql_query += ' drug = \'{}\';'.format(drug)
    logger.debug('calling database with')
    logger.debug('sql_query = {}'.format(sql_query))
    return pandas.read_sql_query(sql_query, engine)


@cached(cache=TTLCache(maxsize=1024, ttl=86400))
def metadata_from_db(species: str, testdate='20190101'):
    """
    retrieves metadata from database
    :param species:
    :param testdate:
    :return: pandas.DataFrame
    """
    logger = logging.getLogger('{}.metadata_from_db()'.format(__name__))
    postgres_str = ('postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'.format(
        username=POSTGRES_USERNAME,
        password=POSTGRES_PASSWORD,
        ipaddress=POSTGRES_ADDRESS,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DBNAME))
    engine = sqlalchemy.create_engine(postgres_str)
    sql_query = 'SELECT * FROM experiment_metadata WHERE testdate > \'{}\' AND species = \'{}\';'.\
        format(
            testdate,
            species)
    logger.debug('calling database with')
    logger.debug('sql_query = {}'.format(sql_query))
    return pandas.read_sql_query(sql_query, engine)


def time_series_from_db(species: str = None, uniqueid: str = None, data_start_date='20190101'):
    """
    pulls time series data from database for species or unique id
    :param species:
    :param uniqueid:
    :param data_start_date:
    :return: pandas.DataFrame
    """
    logger = logging.getLogger('{}.time_series_from_db'.format(__name__))
    if (uniqueid is None) and (species is None):
        raise ValueError('must specify at species or uniqueid')
    postgres_str = ('postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'.format(
        username=POSTGRES_USERNAME,
        password=POSTGRES_PASSWORD,
        ipaddress=POSTGRES_ADDRESS,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DBNAME))
    engine = sqlalchemy.create_engine(postgres_str)
    if uniqueid is None:
        testdate = data_start_date
        stored_data = None
    else:
        testdate = uniqueid.split('__')[1].split('_')[0]
        stored_data = _load_run_data_from_local(uniqueid)
    if not(stored_data is None):
        return stored_data
    sql_query = 'SELECT * FROM experiment_vw WHERE testdate >= \'{}\''.format(testdate)
    if not(species is None):
        sql_query += ' AND species = \'{}\';'.format(species)
    elif not(uniqueid is None):
        sql_query += ' AND uniqueid = \'{}\';'.format(uniqueid)
    logger.debug('calling database with')
    logger.debug('sql_query = {}'.format(sql_query))
    no_data = True
    tries = 0
    try:
        data = pandas.read_sql_query(sql_query, engine)
    except KeyError as err:
        logger.warning('could not read from database, try no. {}: {}'.format(tries, str(err)))
        return None
    if len(data) < 1:
        logger.warning('no data in {}'.format(uniqueid))
        return None
    if not(uniqueid is None):
        _store_run_data_locally(data)
    return data


def _strip_concentration(input_):
    logger = logging.getLogger('{}._strip_concentration()'.format(__name__))
    if isinstance(input_, float):
        return input_
    elif isinstance(input_, int):
        return float(input_)
    elif 'error' in input_.lower():
        return numpy.nan
    elif 'not' in input_.lower():
        return numpy.nan
    elif 'test' in input_.lower():
        return numpy.nan
    elif 'uniq' in input_.lower():
        return numpy.nan
    elif input_ == '':
        return numpy.nan
    elif GTE in input_:
        return float(input_.split(GTE)[1]) * 2
    elif ">" in input_:
        return float(input_.split(">")[1]) * 2
    elif LTE in input_:
        return float(input_.split(LTE)[1]) / 2.0
    elif "<" in input_:
        return float(input_.split("<")[1]) * 2.0
    else:
        try:
            return [float(s) for s in re.findall(r'-?\d+\.?\d*', input_)][0]
        except Exception as err:
            return numpy.nan

    # if len(split_input) == 1:
        # try:
            # return float(split_input[0])
        # except ValueError as err:
            # logger.warning('invalid entry format, ignoring input: {}'.format(str(err)))
            # return numpy.nan
    # else:
        # try:
            # return float(split_input[1])
        # except ValueError as err:
            # logger.warning('invalide entry format, ignoring input: {}'.format(str(err)))
            # return numpy.nan


def resistance_values(mic_data: pandas.DataFrame, save=False):
    """
    extracts float values of drug resistances from mic data
    :param mic_data:
    :param save:    saves file in {DATA_ROOT_DIR}/resistance/<species>.json
    :return: dict
    """
    resistance_dict = {}
    for idx in mic_data.index:
        run_id = mic_data.loc[idx, EndPointsMicData.uniqueid]
        if (resistance_dict.get(run_id) is None):
            run_filter = mic_data[EndPointsMicData.uniqueid] == run_id
            drug_resistance_dict = {}
            for filt_idx in mic_data.index[run_filter]:
                abx = mic_data.loc[filt_idx, EndPointsMicData.abx]
                conc = mic_data.loc[filt_idx, EndPointsMicData.mic_reference]
                drug_resistance_dict.update({abx: _strip_concentration(conc)})
            resistance_dict.update({run_id: drug_resistance_dict})
    if save:
        filename = os.path.join(DATA_ROOT_DIR, 'resistance',
                                '{}.json'.format(mic_data[EndPointsMicData.species].replace('.', '').replace(' ', '')))
        json.dump(resistance_dict, open(filename, 'w'))
    return resistance_dict


def low_pass_filter(vector, order=3, cutoff=1.0):
    nyquist = len(vector) / 2.
    b, a = signal.butter(order, cutoff / nyquist)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, vector, zi=zi*vector[0])
    z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
    return signal.filtfilt(b, a, vector)


def uniform_time(timevector: numpy.array, datavector: numpy.array, t_inc=10):
    """
    refactors data to be on uniform time intervals
    :param timevector:
    :param datavector:
    :param t_inc:
    :return: new_time_vector <numpy.array>, new_data_vector <numpy.array>
    """
    init_time = int(timevector[0])
    final_time = int(timevector[-1] / t_inc) * t_inc
    new_time = numpy.arange(init_time, final_time, t_inc)
    new_data = numpy.zeros(len(new_time))
    for idx in range(len(new_time)):
        old_idx = idx
        while old_idx < len(timevector) and timevector[old_idx] < new_time[idx]:
            old_idx += 1
        if old_idx == len(timevector):
            break
        dt = timevector[old_idx] - new_time[idx]
        if old_idx >= len(timevector) - 2:
            Dd = datavector[old_idx] - datavector[old_idx-1]
            Dt = timevector[old_idx] - timevector[old_idx-1]
            new_data[idx] = datavector[old_idx] + dt * Dd / Dt
        else:
            Dt = timevector[old_idx + 1] - timevector[old_idx]
            Dd = datavector[old_idx + 1] - datavector[old_idx]
            new_data[idx] = datavector[old_idx] + dt * Dd / Dt
    return new_time, new_data


def _get_time_vector(df: pandas.DataFrame, start, end):
    if start > end:
        raise ValueError('starting point occurs after ending point... start={}, end={}'.format(start, end))
    if df is None:
        return None
    df_copy = df.copy()
    spot_values = df_copy[EndPointsTimeSeries.spotid].sort_values().values
    if len(spot_values) < 6:
        return None
    else:
        ref_spot = spot_values[-1]
    spot_filter = df[EndPointsTimeSeries.spotid] == ref_spot
    time_vector = df.loc[df.index[spot_filter], EndPointsTimeSeries.ref_time].unique()
    time_vector.sort()
    if len(time_vector) < end:
        return None
    return time_vector[start: end]


def _get_sorted_spot_list(df, filter_):
    raw_spots_list = df.loc[filter_, EndPointsTimeSeries.spotid].values.tolist()
    spots_to_use = SPOTS_TO_USE.copy()
    #spots_to_use.sort()
    spot_list = []
    for spot in spots_to_use:
        for spot_str in raw_spots_list:
            if (str(spot) in spot_str) and not(spot_str in SPOTS_TO_IGNORE):
                spot_list.append(spot_str)
                break
    return spot_list


def _get_filters(species, testdate, culture, df, well):
    metadata = metadata_from_db(species, testdate=testdate)
    blood_filter = metadata[EndPointsMetadata.source].str.contains(culture)
    df.loc[df.index[df[EndPointsTimeSeries.ignore] == ''], EndPointsTimeSeries.ignore] = False
    do_not_ignore_filter = ~df[EndPointsTimeSeries.ignore]
    species_filter = df[EndPointsTimeSeries.species] == species
    well_filter = df[EndPointsTimeSeries.well] == well
    build_filter = df[EndPointsTimeSeries.librarybuild]
    global_filter = do_not_ignore_filter & blood_filter & build_filter
    return species_filter, well_filter, global_filter


def _get_unique_ids_to_use(metadata: pandas.DataFrame, culture: str):
    not_ignore = ~metadata[EndPointsMetadata.ignore]
    culture_filter = metadata[EndPointsMetadata.source].str.contains(culture)
    build_filter = metadata[EndPointsMetadata.librarybuild]
    filter_ = not_ignore & culture_filter & build_filter
    run_ids = metadata.loc[metadata.index[filter_], EndPointsMetadata.uniqueid].unique()
    return run_ids


def _clean_metadata(metadata: pandas.DataFrame):
    for idx in metadata.index:
        if metadata.loc[idx, EndPointsMetadata.ignore] == 'TRUE':
            metadata.loc[idx, EndPointsMetadata.ignore] = True
        else:
            metadata.loc[idx, EndPointsMetadata.ignore] = False
        if metadata.loc[idx, EndPointsMetadata.source] == '':
            metadata.loc[idx, EndPointsMetadata.source] = 'Neither'
        elif not(isinstance(metadata.loc[idx, EndPointsMetadata.source], str)):
            metadata.loc[idx, EndPointsMetadata.source] = 'Neither'
    return metadata


def generate_heat_map(species: str, drug: str, panel, save_heat_map=True, low_pass=True,
                      culture='Blood', testdate='20190101', run_id_list: list = None,
                      df_in: pandas.DataFrame = None, make_square=True, down_sample=True,
                      start=0, end=MIN_TIME_LEN):
    """
    creates heat maps for all wells for all unique ids after testdate for a given species / drug combination
    :param species:
    :param drug:
    :param panel:
    :param save_heat_map:   saves heatmap as a pickle file in {DATA_ROOT_DIR}/heat_maps/<species>_<drug>.pkl
    :param low_pass:        apply low pass Butterworth filter
    :param culture:
    :param testdate:
    :param run_id_list:     unique ids to use
    :param df_in:           input pandas.DataFrame of time series data
    :param make_square      (bool)
    :param down_sample:     down sample to force square matrix (bool)
    :param start:           starting time index
    :param end:             ending time index
    :return:                dictionary of dictionaries containing 2-D (m x 18) matrices (m is time dimension)
    """
    logger = logging.getLogger('{}.generate_heat_map()'.format(__name__))
    logger.debug('gathering metadata')
    panel_config = load_panel_config(panel)
    if run_id_list is None:
        logger.debug('generating run list')
        metadata = metadata_from_db(species, testdate=testdate)
        metadata = _clean_metadata(metadata)
        run_id_list = _get_unique_ids_to_use(metadata, culture)
    heat_maps = {}
    wells = wells_from_panel_for_drug(panel_config, drug)
    for run_id in run_id_list:
        if df_in is None:
            df = time_series_from_db(uniqueid=run_id)
        else:
            run_filter = df_in[EndPointsTimeSeries.uniqueid] == run_id
            df = df_in.loc[df_in.index[run_filter], :].copy()
        time_vector = _get_time_vector(df, start, end)
        drop_heat_map = False
        if time_vector is None or len(time_vector) < 1:
            logger.debug('there are no data that meet the requirements')
            drop_heat_map = True
            spot_list = []
        else:
            time_filter = df[EndPointsTimeSeries.ref_time] == time_vector[0]
            spot_list = _get_sorted_spot_list(df, time_filter)
        heat_map = {}
        for well in wells:
            if drop_heat_map or time_vector is None or len(time_vector) < 1:
                break
            well_filter = df[EndPointsTimeSeries.well ] == well
            heat_map_arr = numpy.array([])
            for spot in spot_list:
                spot_filter = df[EndPointsTimeSeries.spotid] == spot
                for color in ['ri', 'gi', 'bi']:
                    data_list = []
                    for ts in time_vector:
                        time_filter = df[EndPointsTimeSeries.ref_time] == ts
                        indices = df.index[spot_filter & well_filter & time_filter]
                        color_to_add = df.loc[indices, color].values
                        if len(color_to_add) < 1:
                            drop_heat_map = True
                            break
                        else:
                            data_list.append(df.loc[indices, color].values[0])
                    data_vect = numpy.array(data_list)
                    if len(data_vect) < len(time_vector):
                        drop_heat_map = True
                        break
                    # if len(data_vect) < MIN_TIME_LEN or drop_heat_map:
                    #     logger.warning('time of run less than {} min'.format(MIN_TIME_LEN*10))
                    #     drop_heat_map = True
                    #     break  # break out of color loop
                    _, data_vect = uniform_time(time_vector, data_vect)
                    if low_pass and not drop_heat_map:
                        data_vect = low_pass_filter(data_vect)
                    spot_color_len = 3 * len(spot_list)  # 3 colors x n spots
                    if make_square:
                        vect_to_add = _adjust_vector_dimension(data_vect, spot_color_len, down_sample=down_sample)
                    else:
                        vect_to_add = data_vect[: MIN_TIME_LEN]
                    if vect_to_add is None:
                        drop_heat_map = True
                        break
                    if heat_map_arr.shape[0] < 1 and not drop_heat_map:
                        heat_map_arr = vect_to_add
                    elif drop_heat_map:
                        break
                    else:
                        heat_map_arr = numpy.vstack((heat_map_arr, vect_to_add))
                if drop_heat_map:
                    break  # break out of spot loop
            if drop_heat_map:
                break  # break out of well loop
            else:
                logger.debug('completed heat map for well {} and run {}'.format(well, run_id))
                heat_map.update({well: heat_map_arr.transpose()})
        if drop_heat_map:
            pass  # do not add partial maps for run
        else:
            heat_maps.update({run_id: heat_map})
    if save_heat_map:
        logger.debug('storing heat map')
        filename = os.path.join(DATA_ROOT_DIR, 'heat_maps',
                                '{}_{}_{}_{}.pkl'.format(species.replace('.', '').replace(' ', ''), drug, start, end))
        with open(filename, 'wb') as fid:
            pickle.dump(heat_maps, fid)
    return heat_maps


def _adjust_vector_dimension(old_vector, dim, down_sample=False, end_time_indx=MIN_TIME_LEN):
    """
    sets dimensionality of vector to have dimension 1 x dim
    :param old_vector:
    :param dim:
    :param down_sample:
    :return:
    """
    logger = logging.getLogger('{}._adjust_vector_dimension()'.format(__name__))
    if not(down_sample) and len(old_vector) >= MIN_TIME_LEN:
        return old_vector[MIN_TIME_LEN-dim: MIN_TIME_LEN]
    elif len(old_vector) < MIN_TIME_LEN:
        return None
    #if len(old_vector) % dim > 0:
        #logger.debug('vector does not have a dimension that is a multiple of dim... some data will be removed')
    start = len(old_vector[: end_time_indx]) % dim
    new_vector = numpy.zeros(dim)
    interval = int(len(old_vector[start: end_time_indx]) / dim)
    for idx in range(0, len(old_vector[start: end_time_indx]), interval):
        if idx/interval >= len(new_vector):
            break
        new_vector[int(idx/interval)] = numpy.mean(old_vector[idx: idx+interval])
    return new_vector


def y_from_resistance_dict(run_id: list, resistance_dict: dict, well: str, panel_config: pandas.DataFrame):
    """
    generates the "truth" vector for training
    :param run_id:
    :param resistance_dict:
    :param well:
    :param panel_config:
    :return:    numpy.array
    """
    logger = logging.getLogger('{}.y_from_resistance_dict()'.format(__name__))
    conc, drug = panel_config.loc[well, ['concentration', 'drug']]
    if resistance_dict.get(run_id) is None:
        return None
    elif resistance_dict.get(run_id).get(drug) is None:
        return None
    else:
        res = resistance_dict.get(run_id).get(drug)
        # logger.debug('run id {} well concentration {} resistance {}'.format(run_id, conc, res))
        return int(conc <= res)


def x_vector_from_heat_maps(heat_maps: dict, well, run_id_list: list = None, ):
    """
    generates vector of independent variables to be used in neural network model
    :param heat_maps:
    :param run_id_list:
    :return:   numpy.array
    """
    x_list = []
    run_ids_to_use = []
    heat_map_runs = [run_id for run_id in heat_maps.keys()]
    for run_id in heat_map_runs:
        if run_id_list is None:
            run_ids_to_use.append(run_id)
        elif run_id in run_id_list:
            run_ids_to_use.append(run_id)
    for run_id in run_ids_to_use:
        x_list.append(heat_maps[run_id][well].flatten())
    return numpy.array(x_list), run_ids_to_use


def x_array_from_heat_maps(heat_maps: dict, well, run_id_list: list = None):
    """
    generates m x 1 x n x n matrix of independent variables to be used in convlutional neural network where
    m is the number of measurements, 1 is the number of channels, and the last dimensions are time and spot-color
    respectively
    :param heat_maps:
    :param well:
    :param run_id_list:
    :return:    numpy.array
    """
    x_list = []
    run_ids_to_use = []
    heat_map_runs = [run_id for run_id in heat_maps.keys()]
    for run_id in heat_map_runs:
        if run_id_list is None:
            run_ids_to_use.append(run_id)
        elif run_id in run_id_list:
            run_ids_to_use.append(run_id)
    for run_id in run_ids_to_use:
        heat_map = heat_maps[run_id]
        x_list.append([heat_map[well]])
    return numpy.stack(x_list), run_ids_to_use


def x_y_for_nn(resistance_dict, species, drug, well, panel, heat_maps: dict = None, read_old=False, culture='Blood',
               testdate='20190101', run_id_list: list = None, down_sample=True, nn_type='cnn',
               start=0, end=MIN_TIME_LEN):
    """
    returns independent vector (x), dependent "truth" vector (y), and unique ids to be used in (convolutional)
    neural network models
    :param resistance_dict:
    :param species:
    :param drug:
    :param well:
    :param panel:
    :param heat_maps:
    :param read_old:
    :param culture:
    :param testdate:
    :param run_id_list:
    :param down_sample:
    :param nn_type: 'cnn' or 'nn' used to specify dimension of x
    :return: numpy.array, numpy.array, list
    """
    logger = logging.getLogger('{}.x_y_for_nn()'.format(__name__))
    panel_config = load_panel_config(panel)
    if start is None:
        start = 0
    #todo: add culture to file name of heat map
    if nn_type == 'cnn':
        down_sample = True
        make_square = True
    elif nn_type == 'nn':
        down_sample = False
        make_square = False
    else:
        logger.warning('unknown nn_type using default (cnn)')
        nn_type = 'cnn'
        down_sample = True
        make_square = True
    if read_old:
        filename = os.path.join(DATA_ROOT_DIR, 'heat_maps',
                                '{}_{}.pkl'.format(DATA_ROOT_DIR, species.replace('.', '').replace(' ', ''), drug))
        files = glob.glob(filename)
    else:
        files = []
    if not(heat_maps is None):
        pass
    elif len(files) > 0:
        heat_maps = pickle.load(open(files[0], 'rb'))
    else:
        heat_maps = generate_heat_map(species, drug, panel, culture=culture, testdate=testdate,
                                      make_square=make_square, down_sample=down_sample, start=start,
                                      end=end)
    res_run_ids = [run_id for run_id in resistance_dict.keys()]
    runs_to_use = []
    y_list = []
    if run_id_list is None:
        run_id_list = [run_id for run_id in heat_maps.keys()]

    for run_id in run_id_list:
        if run_id in res_run_ids:
            y = y_from_resistance_dict(run_id, resistance_dict, well, panel_config)
        else:
            y = None
        if y is None:
            pass
        else:
            runs_to_use.append(run_id)
            y_list.append(y)
    if nn_type == 'cnn':
        x, _ = x_array_from_heat_maps(heat_maps, well, run_id_list=runs_to_use)
    else:
        x, _ = x_vector_from_heat_maps(heat_maps, well, run_id_list=runs_to_use)
    return x, numpy.array(y_list), runs_to_use


def train_test_split(x, y, runs, frac=0.7):
    idx_train = numpy.random.choice(numpy.arange(len(y)), int(len(y) * frac), replace=False)
    idx_test = []
    for idx in range(len(y)):
        if not(idx in idx_train):
            idx_test.append(idx)
    return x[idx_train], x[idx_test], y[idx_train], y[idx_test], \
           numpy.array(runs)[idx_train].tolist(), numpy.array(runs)[idx_test].tolist()


def _store_run_data_locally(timeseries_data: pandas.DataFrame):
    filename = os.path.join(DATA_ROOT_DIR, 'timeSeries', '{}.pkl'.format(DATA_ROOT_DIR, timeseries_data.loc[0, EndPointsTimeSeries.uniqueid]))
    pickle.dump(timeseries_data, open(filename, 'wb'))


def _load_run_data_from_local(run_id):
    filename = os.path.join(DATA_ROOT_DIR, 'timeSeries', '{}.pkl'.format(DATA_ROOT_DIR, run_id))
    if os.path.exists(filename):
        return pickle.load(open(filename, 'rb'))
    else:
        return None


@cached(TTLCache(maxsize=512, ttl=86400))
def load_master_file(file_name='20200204_master.xlsx'):
    now = datetime.datetime.now()
    if now.weekday() < 1:
        now -= datetime.timedelta(days=3)
    elif now.weekday() > 5:
        now -= datetime.timedelta(days=now.weekday()-2)
    filename = os.path.join(DATA_ROOT_DIR, 'metadata', file_name)
    return pandas.read_excel(filename)


def load_clsi_data():
    filepath = os.path.join(DATA_ROOT_DIR, 'metadata', 'CLSI.xlsx')
    return pandas.read_excel(filepath)
    #return pandas.read_excel('/Users/ekoch/PycharmProjects/SpecificTechnologies/data/metadata/CLSI.xlsx')


def main():
    """
    for testing purposes and to develop funtions for external use, constaintly evolving pay no attention to this and
    tweak as you see fit
    :return:
    """
    logging.basicConfig(level=logging.DEBUG)
    species = 'E. coli'
    drug = 'Meropenem'
    well = 'H7'
    panel = 'NM43'
    testdate = '20200220'
    resistance_file = glob.glob('{}/resistance/{}.json'.format(DATA_ROOT_DIR,
                                                               species.replace('.', '').replace(' ', '')))
    if len(resistance_file) > 1:
        resistance_dict = json.load(open(resistance_file[0], 'r'))
    else:
        mic_data = mic_data_from_db('E. coli', testdate=testdate)
        resistance_dict = resistance_values(mic_data)
        json.dump(resistance_dict, open('{}/resistance/{}.json'.format(DATA_ROOT_DIR,
                                                                  species.replace('.', '').replace(' ', '')), 'w'))
    #heat_map = generate_heat_map(species, drug, 'NM43', testdate=testdate, make_square=False)
    #run_id_list = [run_id for run_id in heat_map.keys()]
    #x, runs = x_vector_from_heat_maps(heat_map, well, run_id_list=run_id_list)
    x, y, runs = x_y_for_nn(resistance_dict, species, drug, well, panel, heat_maps=None, testdate='20200220',
                            nn_type='nn')
    print('done')


if __name__ == '__main__':
    main()
