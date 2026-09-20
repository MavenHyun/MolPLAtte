"""Precise PDB search: match the MOLECULE NAME, not the whole paper text."""
import json, time, urllib.request, urllib.error

SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
PHRASES = [
    ("T1R sweet/umami",   ["taste receptor type 1"]),
    ("T2R bitter",        ["taste receptor type 2"]),
    ("olfactory receptor",["olfactory receptor"]),
    ("odorant receptor",  ["odorant receptor", "odorant coreceptor"]),
    ("OBP",               ["odorant-binding protein", "odorant binding protein",
                           "pheromone-binding protein"]),
    ("TAAR",              ["trace amine-associated receptor"]),
    ("TRPV1",             ["subfamily V member 1"]),
    ("TRPA1",             ["subfamily A member 1"]),
    ("TRPM8",             ["subfamily M member 8"]),
    ("TRPV3",             ["subfamily V member 3"]),
    ("CaSR",              ["calcium-sensing receptor", "extracellular calcium-sensing"]),
    ("otopetrin sour",    ["otopetrin"]),
    ("PKD2L1 sour",       ["polycystin-2", "polycystic kidney disease 2-like"]),
    ("ENaC salt",         ["sodium channel subunit alpha", "amiloride-sensitive sodium channel"]),
    ("FFAR fat",          ["free fatty acid receptor"]),
    ("chemosensory prot", ["chemosensory protein"]),
    ("gustatory receptor",["gustatory receptor"]),
]

def q_entries(phrase, rows=800):
    nodes=[
      {"type":"terminal","service":"text","parameters":{
        "attribute":"rcsb_polymer_entity.pdbx_description",
        "operator":"contains_phrase","value":phrase}},
      {"type":"terminal","service":"text","parameters":{
        "attribute":"rcsb_entry_info.nonpolymer_entity_count",
        "operator":"greater","value":0}},
    ]
    body={"query":{"type":"group","logical_operator":"and","nodes":nodes},
          "return_type":"entry",
          "request_options":{"paginate":{"start":0,"rows":rows},
                             "results_content_type":["experimental"]}}
    req=urllib.request.Request(SEARCH,data=json.dumps(body).encode(),
                               headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=60) as r:
            if r.status==204: return []
            return [x["identifier"] for x in json.loads(r.read()).get("result_set",[])]
    except urllib.error.HTTPError as e:
        if e.code==204: return []
        print(f"   HTTP {e.code} {phrase!r}"); return []
    except Exception as e:
        print(f"   {type(e).__name__} {phrase!r}"); return []

out={}
for label,phrases in PHRASES:
    ids=set()
    for p in phrases:
        ids |= set(q_entries(p)); time.sleep(0.35)
    out[label]=sorted(ids)
    print(f"  {label:<20}{len(ids):>5}")
json.dump(out,open("/tmp/pdb_hits2.json","w"))
print(f"\ntotal distinct: {len({i for v in out.values() for i in v})}")
