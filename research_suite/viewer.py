import base64
import json
from pathlib import Path
import numpy as np
from imaging import normalize_display


def build_viewer(case,modality,prediction=None):
    image=case['images'][:,modality].transpose(2,1,0)
    encoded=np.rint(normalize_display(image,1,99)*255).astype(np.uint8)
    def pack(a):return base64.b64encode(np.ascontiguousarray(a,dtype=np.uint8).tobytes()).decode('ascii') if a is not None else ''
    payload={'shape':list(image.shape),'spacing':case['spacing'].tolist(),'image':pack(encoded),
             'mask':pack(prediction),'reference':pack(case['reference']),'labels':[1,2,3]}

    return Path(__file__).with_name('viewer.html').read_text(encoding='utf-8').replace('__VOLUME_PAYLOAD__',json.dumps(payload))
