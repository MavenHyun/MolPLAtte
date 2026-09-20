"""Construct taste_odor_pdb.json entries for the new PDB structures."""
import json, re, collections

rows = json.load(open("/tmp/pdb_new_meta.json"))
hits = json.load(open("/tmp/pdb_hits2.json"))
label = {}
for lab, ids in hits.items():
    for i in ids: label.setdefault(i.upper(), lab)

# EXCLUDED per instruction: fat taste (FFAR1/FFAR4 -- antidiabetic targets).
EXCLUDE = {"FFAR fat"}

FAMILY = {
    "TRPV3":             ("TRPV3 – warmth/camphor receptor (chemesthesis)", "taste"),
    "TRPV1":             ("TRPV1 – capsaicin/vanilloid receptor (chemesthesis)", "taste"),
    "TRPA1":             ("TRPA1 – irritant/pungency receptor (chemesthesis)", "taste"),
    "TRPM8":             ("TRPM8 – cooling/menthol receptor (chemesthesis)", "taste"),
    "CaSR":              ("CaSR – kokumi/calcium-sensing receptor", "taste"),
    "PKD2L1 sour":       ("OTOP1/PKD2L1 – sour taste channel", "taste"),
    "TAAR":              ("TAAR – olfactory amine receptor", "odor"),
    "OBP":               ("OBP – insect odorant/pheromone binding protein", "odor"),
    "odorant receptor":  ("Odorant receptor (insect OR/Orco/IR)", "odor"),
    "olfactory receptor":("Olfactory receptor (vertebrate GPCR)", "odor"),
    "T1R sweet/umami":   ("T1R – sweet/umami receptor", "taste"),
    "T2R bitter":        ("T2R – bitter taste receptor", "taste"),
    "gustatory receptor":("Gustatory receptor (insect GR)", "taste"),
    "chemosensory prot": ("CSP – chemosensory protein", "odor"),
    "ENaC salt":         ("ENaC – salt taste channel", "taste"),
}

# families whose deposited ligands are pharmaceutical rather than flavour
PHARMA_LIGANDS = {"TAAR"}

JUNK = {"PEG","PG0","PG4","PG6","1PE","2PE","P6G","EDO","GOL","MPD","DMS","SO4","PO4","ACT",
        "CIT","TRS","EPE","NAG","BMA","MAN","CLR","OLC","PLM","OLA","Y01","LMT","LMN","HOH",
        "CA","NA","K","CL","MG","ZN","ATP","ADP","GDP","GTP","GNP","AF3","BEF","CHS","POV",
        "D10","UNL","IPA","FMT","NO3","F09","LBN","PC1","PEE","PTY","SQD","HEM","NDG","BGC"}
JUNKNAME = re.compile(r"glycol|glycerol|phosphatidyl|phosphocholine|monoolein|cholesterol|"
                      r"maltoside|digitonin|lauryl|decyl|nonyl|octyl", re.I)

out, skipped = [], collections.Counter()
for e in rows:
    pid = e["rcsb_id"].upper()
    lab = label.get(pid)
    if lab in EXCLUDE:
        skipped["excluded_fat_taste"] += 1; continue
    if lab not in FAMILY:
        skipped[f"no_family_map:{lab}"] += 1; continue
    ri = e.get("rcsb_entry_info") or {}
    res = (ri.get("resolution_combined") or [None])[0]
    if not res or res > 3.5:
        skipped["resolution"] += 1; continue

    ligs = []
    for ne in (e.get("nonpolymer_entities") or []):
        cc = ((ne.get("nonpolymer_comp") or {}).get("chem_comp") or {})
        cid, nm, mw = cc.get("id"), cc.get("name") or "", cc.get("formula_weight") or 0
        if cid and cid not in JUNK and not JUNKNAME.search(nm) and 100 <= mw <= 900:
            ligs.append([cid, nm, mw])
    if not ligs:
        skipped["no_real_ligand"] += 1; continue

    pes = e.get("polymer_entities") or []
    desc = [((pe.get("rcsb_polymer_entity") or {}).get("pdbx_description") or "") for pe in pes]
    orgs = sorted({o["ncbi_scientific_name"] for pe in pes
                   for o in (pe.get("rcsb_entity_source_organism") or [])
                   if o.get("ncbi_scientific_name")})
    unis = sorted({r_["database_accession"] for pe in pes
                   for r_ in ((pe.get("rcsb_polymer_entity_container_identifiers") or {})
                              .get("reference_sequence_identifiers") or [])
                   if r_.get("database_name") == "UniProt"})
    fam, cat = FAMILY[lab]
    out.append({
        "id": pid,
        "title": ((e.get("struct") or {}).get("title") or "").strip(),
        "families": [fam], "cats": [cat],
        "ligands": ligs, "n_lig": len(ligs),
        "cognate_ligands": ligs, "n_cognate": len(ligs),
        "n_polymer_entities": len(pes),
        "desc": desc, "organisms": orgs, "uniprot": unis,
        "method": ri.get("experimental_method") or "",
        "resolution": res, "released": "",
        "matched_terms": [lab],
        "tier": "T1_ligand_complex",
        "source": "pdb_search_2026-09-20",
        "ligand_class": "pharmaceutical" if lab in PHARMA_LIGANDS else "flavour_plausible",
    })

json.dump(out, open("/tmp/new_entries.json", "w"), indent=1)
print(f"new entries built: {len(out)}")
print("skipped:", dict(skipped))
byf = collections.Counter(x["families"][0] for x in out)
print(f"\n{'family':<52}{'entries':>8}  ligand class")
for f, n in byf.most_common():
    lc = {x["ligand_class"] for x in out if x["families"][0] == f}
    print(f"  {f[:50]:<52}{n:>6}  {'/'.join(sorted(lc))}")
print(f"\nligands to download: {len({l[0] for x in out for l in x['ligands']})}")
