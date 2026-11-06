"""Explicit RGB quality/sampling contracts independent of 20Hz controls."""


DEFAULT_CAPTURE = {'schema': 'rgb-capture-v1', 'width': 1920, 'height': 1080, 'fps': 20,
                   'control_fps': 20, 'crf': 23, 'preset': 'medium'}


def capture_contract(spec=None):
    result = dict(DEFAULT_CAPTURE if spec is None else spec)
    if set(result) != set(DEFAULT_CAPTURE) or result['schema'] != 'rgb-capture-v1':
        raise ValueError('Unsupported RGB capture contract')
    if (result['control_fps'] != 20 or type(result['fps']) is not int
            or result['fps'] not in (1, 2, 4, 5, 10, 20)):
        raise ValueError('RGB fps must divide the 20Hz control frequency')
    for key in ('width', 'height'):
        if type(result[key]) is not int or not 64 <= result[key] <= 4096 or result[key] % 2:
            raise ValueError('RGB dimensions must be positive even dimensions from 64 to 4096')
    if type(result['crf']) is not int or not 0 <= result['crf'] <= 51:
        raise ValueError('Invalid RGB CRF')
    if result['preset'] not in ('ultrafast', 'superfast', 'veryfast', 'fast', 'medium'):
        raise ValueError('Unsupported x264 speed preset')
    return result


def capture_from_config(config):
    profile = config.get('capture_profile')
    if profile is None:
        return None
    if profile != 'speed-v1':
        raise ValueError('Unknown capture profile')
    return capture_contract({'schema': 'rgb-capture-v1', 'width': config['rgb_width'],
        'height': config['rgb_height'], 'fps': config['rgb_fps'], 'control_fps': 20,
        'crf': config.get('rgb_crf', 28), 'preset': config.get('rgb_preset', 'ultrafast')})


def recorded_rgb_contract(arrays, online_contract):
    spec = online_contract.get('rgb_capture')
    count = len(arrays['timestamp'])
    if spec is None:
        return count, None  # Frozen legacy 1080p20 records.
    spec = capture_contract(spec)
    indices = list(range(0, count, 20 // spec['fps']))
    recorded = online_contract.get('rgb_control_indices')
    if (not isinstance(recorded, list) or any(type(i) is not int for i in recorded)
            or recorded != indices):
        raise ValueError('RGB indices do not cover the declared control-step sampling')
    return len(indices), spec
