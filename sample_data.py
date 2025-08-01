from collections import defaultdict
import numpy as np

sample_data = {'info': defaultdict(list,
             {'name': ['EEG'],
              'type': ['EEG'],
              'channel_count': ['34'],
              'channel_format': ['float32'],
              'source_id': ['PRO_073'],
              'nominal_srate': ['1000.000000000000'],
              'version': ['1.100000000000000'],
              'created_at': ['1031834.531415200'],
              'uid': ['4e4b9131-7584-47c6-8087-ac75059db44a'],
              'session_id': ['default'],
              'hostname': ['CS3024D00078917'],
              'v4address': [None],
              'v4data_port': ['16573'],
              'v4service_port': ['16573'],
              'v6address': [None],
              'v6data_port': ['16573'],
              'v6service_port': ['16573'],
              'desc': [defaultdict(list,
                           {'manufacturer': ['mBrainTrain'],
                            'cap': [defaultdict(list,
                                         {'name': ['EasyCap'],
                                          'size': ['54'],
                                          'labelscheme': ['10-20'],
                                          'filetype': ['sfp']})],
                            'channels': [defaultdict(list,
                                         {'type': ['EEG'],
                                          'channel': [defaultdict(list,
                                                       {'label': ['Fp1'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['-27.0'],
                                                                      'Y': ['86.0'],
                                                                      'Z': ['36.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['Fp2'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['27.0'],
                                                                      'Y': ['86.0'],
                                                                      'Z': ['36.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['F3'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['-47.0'],
                                                                      'Y': ['62.0'],
                                                                      'Z': ['80.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['F4'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['47.0'],
                                                                      'Y': ['62.0'],
                                                                      'Z': ['80.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['C3'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['-61.0'],
                                                                      'Y': ['0.0'],
                                                                      'Z': ['97.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['C4'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['61.0'],
                                                                      'Y': ['0.0'],
                                                                      'Z': ['97.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['P3'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['-47.0'],
                                                                      'Y': ['-62.0'],
                                                                      'Z': ['80.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['P4'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['47.0'],
                                                                      'Y': ['-62.0'],
                                                                      'Z': ['80.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['O1'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['-27.0'],
                                                                      'Y': ['-86.0'],
                                                                      'Z': ['36.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['O2'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['27.0'],
                                                                      'Y': ['-86.0'],
                                                                      'Z': ['36.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['F7'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['-67.0'],
                                                                      'Y': ['52.0'],
                                                                      'Z': ['36.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['F8'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['67.0'],
                                                                      'Y': ['52.0'],
                                                                      'Z': ['36.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['T7'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['-78.0'],
                                                                      'Y': ['0.0'],
                                                                      'Z': ['36.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['T8'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['78.0'],
                                                                      'Y': ['0.0'],
                                                                      'Z': ['36.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['P7'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['-67.0'],
                                                                      'Y': ['-52.0'],
                                                                      'Z': ['36.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['P8'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['67.0'],
                                                                      'Y': ['-52.0'],
                                                                      'Z': ['36.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['Fz'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['0.0'],
                                                                      'Y': ['67.0'],
                                                                      'Z': ['95.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['Cz'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['0.0'],
                                                                      'Y': ['0.0'],
                                                                      'Z': ['120.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['Pz'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['0.0'],
                                                                      'Y': ['-67.0'],
                                                                      'Z': ['95.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['M1'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['-73.0'],
                                                                      'Y': ['-25.0'],
                                                                      'Z': ['0.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['M2'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['73.0'],
                                                                      'Y': ['-25.0'],
                                                                      'Z': ['0.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['AFz'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['0.0'],
                                                                      'Y': ['83.0'],
                                                                      'Z': ['69.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['CPz'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['0.0'],
                                                                      'Y': ['-34.0'],
                                                                      'Z': ['113.00000000000001']})]}),
                                           defaultdict(list,
                                                       {'label': ['POz'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['0.0'],
                                                                      'Y': ['-83.0'],
                                                                      'Z': ['69.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['AccX'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['0.0'],
                                                                      'Y': ['0.0'],
                                                                      'Z': ['0.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['AccY'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['0.0'],
                                                                      'Y': ['0.0'],
                                                                      'Z': ['0.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['AccZ'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['0.0'],
                                                                      'Y': ['0.0'],
                                                                      'Z': ['0.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['GyroX'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['0.0'],
                                                                      'Y': ['0.0'],
                                                                      'Z': ['0.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['GyroY'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['0.0'],
                                                                      'Y': ['0.0'],
                                                                      'Z': ['0.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['GyroZ'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['0.0'],
                                                                      'Y': ['0.0'],
                                                                      'Z': ['0.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['QuarW'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['0.0'],
                                                                      'Y': ['0.0'],
                                                                      'Z': ['0.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['QuarX'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['0.0'],
                                                                      'Y': ['0.0'],
                                                                      'Z': ['0.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['QuarY'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['0.0'],
                                                                      'Y': ['0.0'],
                                                                      'Z': ['0.0']})]}),
                                           defaultdict(list,
                                                       {'label': ['QuarZ'],
                                                        'location': [defaultdict(list,
                                                                     {'X': ['0.0'],
                                                                      'Y': ['0.0'],
                                                                      'Z': ['0.0']})]})]})],
                            'timecreated': ['2023-03-27 16:03:35.523']})],
              'stream_id': 2,
              'effective_srate': np.float64(1000.0084635909244),
              'segments': [(np.int64(0), np.int64(44903))]}),
 'footer': {'info': defaultdict(list,
              {'first_timestamp': ['1033126.7430806706'],
               'last_timestamp': ['1033171.6457019945'],
               'sample_count': ['44904.0'],
               'clock_offsets': [defaultdict(list,
                            {'time': ['1033127.3956516',
                              '1033132.39666725',
                              '1033137.3997790499',
                              '1033142.40178855',
                              '1033147.4028729999',
                              '1033152.4037608501',
                              '1033157.40903575',
                              '1033162.4096248499',
                              '1033167.4109703001',
                              '1033172.4117749'],
                             'value': ['-1.1299969628453255E-5',
                              '2.83500412479043E-5',
                              '-1.044996315613389E-5',
                              '-1.4549994375556707E-5',
                              '-1.869996776804328E-5',
                              '4.249974153935909E-6',
                              '-2.0949984900653362E-5',
                              '-1.6499543562531471E-6',
                              '-4.100031219422817E-6',
                              '1.0700023267418146E-5']})]})},
 'time_series': np.array([[ 4.7313726e+03, -4.7048184e+02,  4.9991465e+03, ...,
          4.6203613e-01, -7.1337891e-01,  2.3675537e-01],
        [ 4.7131782e+03, -4.8518930e+02,  5.0005547e+03, ...,
          4.6203613e-01, -7.1337891e-01,  2.3675537e-01],
        [ 4.6802539e+03, -5.1831458e+02,  4.9420151e+03, ...,
          4.6203613e-01, -7.1337891e-01,  2.3675537e-01],
        ...,
        [ 4.3297788e+03, -6.0291595e+02,  4.9466421e+03, ...,
          4.6032715e-01, -7.0959473e-01,  2.4487305e-01],
        [ 4.2950664e+03, -6.3686823e+02,  4.8873877e+03, ...,
          4.6032715e-01, -7.0959473e-01,  2.4487305e-01],
        [ 4.3130820e+03, -6.2365839e+02,  4.8858901e+03, ...,
          4.6032715e-01, -7.0959473e-01,  2.4487305e-01]], dtype=np.float32),
 'time_stamps': np.array([1033126.74307156, 1033126.74407155, 1033126.74507154, ...,
        1033171.64369153, 1033171.64469152, 1033171.64569151]),
 'clock_times': [1033127.3956516,
  1033132.39666725,
  1033137.3997790499,
  1033142.40178855,
  1033147.4028729999,
  1033152.4037608501,
  1033157.40903575,
  1033162.4096248499,
  1033167.4109703001,
  1033172.4117749],
 'clock_values': [-1.1299969628453255e-05,
  2.83500412479043e-05,
  -1.044996315613389e-05,
  -1.4549994375556707e-05,
  -1.869996776804328e-05,
  4.249974153935909e-06,
  -2.0949984900653362e-05,
  -1.6499543562531471e-06,
  -4.100031219422817e-06,
  1.0700023267418146e-05]}