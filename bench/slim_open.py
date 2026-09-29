"""Does Google's runtime OPEN a slim .litertlm made by litertlm_slim.py?

The slim file drops `tf_lite_embedder` (the token embedding is also the LM head inside
`tf_lite_prefill_decode`) and the three audio sections. Google's runtime has no code to read the
embedding from the LM head, so it is EXPECTED to fail once it needs the text embedding. What this
checks is WHERE it fails: the container must be valid, i.e. the header parses and every section maps,
so the failure must be about the missing embedder, not about the file. The full file is the control:
the same prompt must be answered.

Usage: slim_open.py <full.litertlm> <slim.litertlm>
"""
import os
import sys
import time

import litert_lm as lm


def text_of(resp):
    c = resp.get("content", resp) if isinstance(resp, dict) else resp
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(x.get("text", "") for x in c if isinstance(x, dict))
    return str(resp)


def trial(label, path):
    cache = os.path.abspath(".lrtcache_" + label)
    os.makedirs(cache, exist_ok=True)
    t0 = time.time()
    try:
        eng = lm.Engine(path, backend=lm.Backend.CPU(), max_num_tokens=2048, cache_dir=cache)   # 2048: the 1024-position prefill must fit the KV (512 failed in DYNAMIC_UPDATE_SLICE)
    except Exception as e:
        print("%s: ENGINE CREATE FAILED after %.0f s: %s" % (label, time.time() - t0, e), flush=True)
        return
    print("%s: engine created in %.0f s" % (label, time.time() - t0), flush=True)
    try:
        conv = eng.create_conversation(sampler_config=lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0),
                                       max_output_tokens=24)
        reply = text_of(conv.send_message("Say exactly the following sentence and nothing else: the ferry leaves at six"))
        print("%s: REPLY %r" % (label, reply), flush=True)
    except Exception as e:
        print("%s: SEND FAILED: %s" % (label, e), flush=True)


trial("full", sys.argv[1])
trial("slim", sys.argv[2])
