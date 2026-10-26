"""Shared normalisation library (country-agnostic, no external data).

Every record (S1/S2/S3, any split) -> a fixed set of clean fields that all blocking
methods and the matcher use:

  name fields : n_full, n_core, n_alt (other side of a DBA), n_nospace, n_skel,
                legal (canonical legal-form tokens), flags (is_domain, is_hashtag, has_dba, was_indic)
  addr fields : a_full, a_street, a_num (primary house number), a_nums (all numbers),
                a_city, a_locs (all non-street place segments), a_state (canonical), a_empty

Only hand-written domain knowledge (abbreviation lists, state codes, legal forms) and a
token map learned from the TRAINING pairs (learn_translit.py) are used.
"""
import os
import re
import json
import unicodedata
from functools import lru_cache

from unidecode import unidecode

from config import WORK_DIR

# ============================================================== dictionaries
LEGAL_CANON = {
    # English / US / India
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "limited": "ltd", "ltd": "ltd", "llc": "llc", "llp": "llp",
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "plc": "plc", "pllc": "pllc", "lp": "lp", "pc": "pc", "pa": "pa", "pbc": "pbc", "opc": "opc",
    "ltda": "ltd",
    # France
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci", "snc": "snc",
    "ei": "ei", "scop": "scop", "gie": "gie", "selarl": "selarl", "scm": "scm", "sca": "sca",
    "ets": "ets", "etablissements": "ets", "etablissement": "ets", "cie": "co", "compagnie": "co",
    # German-ish forms seen in the data
    "gmbh": "gmbh", "ag": "ag", "aktiengesellschaft": "ag",
}
STOP_NAME = {"the", "and", "of", "a", "an", "at", "for", "in", "on",
             "de", "du", "des", "la", "le", "les", "et", "d", "l", "en"}
HONORIFICS = {"mr", "mrs", "ms", "dr", "smt", "shri", "sri", "shree", "shrimati", "m", "s", "messrs", "ms."}
NAME_ABBR = {"intl": "international", "natl": "national", "mfg": "manufacturing", "svc": "services",
             "svcs": "services", "mgmt": "management", "assoc": "associates", "assocs": "associates",
             "bros": "brothers", "ent": "enterprises", "entp": "enterprises", "sys": "systems", "grp": "group", "univ": "university",
             "hosp": "hospital", "ctr": "center", "centre": "center", "dept": "department",
             "st": "saint", "ste": "sainte", "shree": "shri", "sri": "shri", "shrí": "shri"}

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "dc": "district of columbia", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland", "ma": "massachusetts",
    "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri", "mt": "montana",
    "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico",
    "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina", "sd": "south dakota",
    "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "pr": "puerto rico",
}
IN_STATES = {
    "mh": "maharashtra", "dl": "delhi", "up": "uttar pradesh", "ka": "karnataka", "tn": "tamil nadu",
    "wb": "west bengal", "gj": "gujarat", "tg": "telangana", "ts": "telangana", "hr": "haryana",
    "rj": "rajasthan", "kl": "kerala", "mp": "madhya pradesh", "br": "bihar", "ap": "andhra pradesh",
    "pb": "punjab", "od": "odisha", "or": "odisha", "orissa": "odisha", "cg": "chhattisgarh",
    "ct": "chhattisgarh", "jh": "jharkhand", "uk": "uttarakhand", "ut": "uttarakhand",
    "uttaranchal": "uttarakhand", "hp": "himachal pradesh", "ga": "goa", "as": "assam",
    "jk": "jammu and kashmir", "ch": "chandigarh", "py": "puducherry", "pondicherry": "puducherry",
    "nct of delhi": "delhi", "new delhi": None,  # 'new delhi' is a city, not a state
}
FR_ADMIN = {  # department -> region (hand-written; only admin areas are compared, never used as lookup)
    "nord": "hauts de france", "pas de calais": "hauts de france", "somme": "hauts de france",
    "aisne": "hauts de france", "oise": "hauts de france",
    "gironde": "nouvelle aquitaine", "landes": "nouvelle aquitaine", "charente maritime": "nouvelle aquitaine",
    "pyrenees atlantiques": "nouvelle aquitaine", "dordogne": "nouvelle aquitaine",
    "loire atlantique": "pays de la loire", "vendee": "pays de la loire", "maine et loire": "pays de la loire",
    "sarthe": "pays de la loire", "mayenne": "pays de la loire",
    "ile de france": "ile de france", "paris": None,
}
STATE_CANON = {}
for d in (US_STATES, IN_STATES):
    for k, v in d.items():
        if v:
            STATE_CANON[k] = v
            STATE_CANON[v] = v
for k, v in FR_ADMIN.items():
    if v:
        STATE_CANON[k] = v
        STATE_CANON[v] = v
for r in ("hauts de france", "nouvelle aquitaine", "pays de la loire", "bretagne", "normandie",
          "occitanie", "grand est", "auvergne rhone alpes", "provence alpes cote d azur",
          "bourgogne franche comte", "centre val de loire", "corse"):
    STATE_CANON[r] = r
STATE_CODES_US = set(US_STATES)          # 2-letter codes are ambiguous across countries -> resolved by country
STATE_CODES_IN = {k for k in IN_STATES if len(k) == 2}

CITY_ALIAS = {  # common official renames / spellings (India) + 'saint' forms (France)
    "bengaluru": "bangalore", "gurugram": "gurgaon", "ahmadabad": "ahmedabad", "bombay": "mumbai",
    "calcutta": "kolkata", "madras": "chennai", "trivandrum": "thiruvananthapuram", "mysuru": "mysore",
    "baroda": "vadodara", "poona": "pune", "cochin": "kochi", "vizag": "visakhapatnam",
    "benares": "varanasi", "allahabad": "prayagraj", "gautam buddha nagar": "gautambuddhanagar",
}

STREET_TYPES = {  # -> canonical long form
    "st": "street", "str": "street", "street": "street", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "avenue": "avenue", "rd": "road", "road": "road", "dr": "drive", "drv": "drive", "drive": "drive",
    "ln": "lane", "lane": "lane", "ct": "court", "crt": "court", "court": "court", "pl": "place",
    "place": "place", "blvd": "boulevard", "bd": "boulevard", "bvd": "boulevard", "boul": "boulevard",
    "boulevrd": "boulevard", "boulevard": "boulevard", "cir": "circle", "circle": "circle", "trl": "trail",
    "trail": "trail", "pkwy": "parkway", "parkway": "parkway", "hwy": "highway", "highway": "highway",
    "ter": "terrace", "terr": "terrace", "terrace": "terrace", "way": "way", "sq": "square",
    "square": "square", "cv": "cove", "cove": "cove", "pt": "point", "loop": "loop", "run": "run",
    "xing": "crossing", "crossing": "crossing", "aly": "alley", "alley": "alley", "path": "path",
    "pass": "pass", "row": "row", "walk": "walk", "plz": "plaza", "plaza": "plaza", "expy": "expressway",
    "fwy": "freeway", "tpke": "turnpike", "rte": "route", "route": "route", "mall": "mall", "hts": "heights",
    "mnr": "manor", "crk": "creek", "rdg": "ridge", "vw": "view", "grv": "grove", "hl": "hill",
    "hls": "hills", "lk": "lake", "mdw": "meadow", "mdws": "meadows", "pnes": "pines", "spg": "spring",
    "spgs": "springs", "vly": "valley", "cyn": "canyon", "est": "estate", "ests": "estates",
    # France
    "rue": "rue", "r": "rue", "imp": "impasse", "impasse": "impasse", "all": "allee", "allee": "allee",
    "aelee": "allee", "ch": "chemin", "chem": "chemin", "chemin": "chemin", "quai": "quai", "qu": "quai",
    "cours": "cours", "crs": "cours", "fbg": "faubourg", "faubourg": "faubourg", "pte": "porte",
    "porte": "porte", "prom": "promenade", "promenade": "promenade", "rpt": "rondpoint", "sente": "sente",
    "villa": "villa", "cite": "cite", "hameau": "hameau", "lieu dit": "lieudit", "residence": "residence",
    "res": "residence", "bat": "batiment", "batiment": "batiment",
}
DIRECTIONS = {"n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
              "se": "southeast", "sw": "southwest", "north": "north", "south": "south", "east": "east",
              "west": "west"}
ADDR_WORDS = {"nr": "near", "opp": "opposite", "opposite": "opposite", "bldg": "building",
              "blk": "block", "sec": "sector", "ngr": "nagar", "mkt": "market", "stn": "station",
              "dist": "district", "clny": "colony", "chwk": "chowk", "flr": "floor", "fl": "floor",
              "grnd": "ground", "gf": "ground floor", "ff": "first floor", "mg": "mahatma gandhi",
              "twp": "", "cdp": "", "city of": "", "township": ""}
UNIT_RE = re.compile(r"^(unit|apt|apartment|ste|suite|#|bldg|building|floor|fl|room|rm|po box|p o box|"
                     r"pobox|box|dept|lot|space|spc|trlr|pmb|batiment|bat|etage|appt|rdc)\b")
NUM_LABELS = {"no", "nos", "number", "num", "plot", "flat", "door", "house", "h", "hn", "hno", "kh", "khasra",
              "sr", "survey", "sy", "ward", "shop", "office", "n", "d", "dno", "hig", "lig", "mig", "bis", "ter",
              "c", "o", "co", "s"}
PLACEHOLDER_RE = re.compile(r"^\s*(<\s*null\s*>|null|none|nan|n/?a|na|-+|\.+|unknown|not available)\s*$", re.I)

LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b"})

INDIC_RE = re.compile(r"[ऀ-෿]")
ZW_RE = re.compile(r"[​-‍﻿­]")
WS_RE = re.compile(r"\s+")
DBA_RE = re.compile(r"\s+(?:d\s*/\s*b\s*/\s*a|dba|t\s*/\s*a|a\.?k\.?a\.?|doing business as|trading as)\s*:?\s+", re.I)
PIPE_WWW_RE = re.compile(r"\s*\|\s*(www\.)?\S+\.\S+\s*$", re.I)
DOMAIN_RE = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:c[o0]m|co\.in|co|in|net|org|fr|biz|info|io)\b", re.I)
STORE_NUM_RE = re.compile(r"#\s*\d+\b")
LEET_RE = re.compile(r"(?<=[a-z])[0-8](?=[a-z])|(?<=[a-z]{3})[01](?=\b)|(?<=\b)[01](?=[a-z]{3})")

# ============================================================== translit map (learned from train)
_TMAP = None


def _translit_maps():
    global _TMAP
    if _TMAP is None:
        p = os.path.join(WORK_DIR, "translit_map.json")
        if os.path.exists(p):
            m = json.load(open(p, encoding="utf-8"))
            seg = {k.strip().strip('"').strip(): v for k, v in m["segment"].items()}
            # keep only segment mappings that are state names (drops noisy learned entries)
            seg = {k: v for k, v in seg.items() if v in STATE_CANON}
            _TMAP = (m["token"], seg)
        else:
            _TMAP = ({}, {})
    return _TMAP


@lru_cache(maxsize=200_000)
def translit_token(tok: str) -> str:
    """Indic token -> Latin: learned map first, then unidecode + phonetic clean-up."""
    tmap, _ = _translit_maps()
    tok = ZW_RE.sub("", tok)
    if tok in tmap:
        return tmap[tok]
    s = unidecode(tok).lower()
    s = re.sub(r"(aa|ee|ii|oo|uu)", lambda m: m.group(0)[0], s)
    s = s.replace("n'", "n").replace("~", "")
    return re.sub(r"[^a-z0-9]", "", s)


def translit_text(s: str) -> str:
    if not INDIC_RE.search(s):
        return s
    out = []
    for t in re.split(r"(\s+|[,()\-/.])", s):
        out.append(translit_token(t) if INDIC_RE.search(t) else t)
    return "".join(out)


# ============================================================== helpers
def ascii_lower(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = ZW_RE.sub("", s)
    return unidecode(s).lower()


@lru_cache(maxsize=500_000)
def skeleton(s: str) -> str:
    """Consonant skeleton: robust to vowel / transliteration noise ('raam maarketing' ~ 'ram marketing')."""
    out = []
    for w in s.split():
        w = re.sub(r"(.)\1+", r"\1", w)          # collapse doubles
        w = w[0] + re.sub(r"[aeiouy]", "", w[1:]) if w else w
        w = w.replace("ph", "f").replace("v", "w").replace("z", "s").replace("q", "k").replace("c", "k")
        out.append(w)
    return " ".join(out)


def _dedupe_consecutive(toks):
    out = []
    for t in toks:
        if not out or out[-1] != t:
            out.append(t)
    return out


# ============================================================== names
def normalize_name(raw: str) -> dict:
    s = raw or ""
    s = unicodedata.normalize("NFKC", s)
    s = ZW_RE.sub("", s)
    was_indic = bool(INDIC_RE.search(s))
    if was_indic:
        s = translit_text(s)
    s = PIPE_WWW_RE.sub("", s)
    has_dba = False
    alt = ""
    m = DBA_RE.search(s)
    if m:
        has_dba = True
        left, right = s[:m.start()], s[m.end():]
        s, alt = right, left                      # the 'real' name is usually after the DBA marker
    is_hashtag = "#" in s and bool(re.search(r"#[A-Za-z]", s))
    s = STORE_NUM_RE.sub(" ", s)
    s = unidecode(s).lower()

    is_domain = False
    tokens_raw = s.split()
    new = []
    for t in tokens_raw:
        dm = DOMAIN_RE.match(t.strip("()[]{}<>|*@#"))
        if dm:
            is_domain = True
            new.append(dm.group(1))
        else:
            new.append(t)
    s = " ".join(new)

    full_toks = _name_tokens(s)
    alt_toks = _name_tokens(unidecode(alt).lower()) if alt else []
    core = [t for t in full_toks if t not in LEGAL_CANON.values() and t not in STOP_NAME and t not in HONORIFICS]
    if not core:
        core = [t for t in full_toks if t not in STOP_NAME] or full_toks
    alt_core = [t for t in alt_toks if t not in LEGAL_CANON.values() and t not in STOP_NAME and t not in HONORIFICS]
    legal = sorted({t for t in full_toks if t in set(LEGAL_CANON.values())})
    n_core = " ".join(core)
    return {
        "n_full": " ".join(full_toks),
        "n_core": n_core,
        "n_alt": " ".join(alt_core),
        "n_nospace": n_core.replace(" ", ""),
        "n_skel": skeleton(n_core),
        "legal": " ".join(legal),
        "is_domain": is_domain,
        "is_hashtag": is_hashtag,
        "has_dba": has_dba,
        "was_indic": was_indic,
    }


def _name_tokens(s: str):
    s = s.replace("&", " and ").replace("+", " and ").replace("@", " ")
    s = re.sub(r"\b([a-z])\.(?=[a-z]\.)", r"\1", s)          # l.l.c. -> llc. ; p.a. -> pa.
    s = re.sub(r"(?<=[a-z])\.(?=[a-z]\b)", "", s)
    s = s.replace("'", "").replace("`", "")
    s = LEET_RE.sub(lambda m: m.group(0).translate(LEET), s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = []
    for t in s.split():
        t = NAME_ABBR.get(t, t)
        t = LEGAL_CANON.get(t, t)
        toks.extend(t.split())
    # 'pvt ltd' duplicates, 'smart smart' -> dedupe consecutive
    return _dedupe_consecutive(toks)


# ============================================================== addresses
def _canon_state(seg: str, country: str):
    """Return canonical state/region for a segment if it is one, else None."""
    c = country.lower()
    if len(seg) == 2:
        if c == "us" and seg in STATE_CODES_US:
            return US_STATES[seg]
        if c == "india" and seg in STATE_CODES_IN:
            return IN_STATES[seg]
        return None
    return STATE_CANON.get(seg)


def _canon_addr_tokens(seg: str, country: str):
    c = country.lower()
    seg = re.sub(r"\bst\s*\.?\s*-\s*", "saint ", seg) if c == "france" else seg
    seg = seg.replace("n°", " no ").replace("#", " ")
    seg = re.sub(r"[^a-z0-9/ ]+", " ", seg)
    toks = seg.split()
    out = []
    n = len(toks)
    for i, t in enumerate(toks):
        if c == "france" and t in ("st", "ste"):
            out.append("saint"); continue
        if c == "us" and t == "saint" and i == n - 1:
            out.append("street"); continue           # noise: 'ST' expanded to 'SAINT' at end of street
        if t in STREET_TYPES and (c != "india" or t not in ("r", "all", "ch", "est", "pt")):
            out.append(STREET_TYPES[t]); continue
        if t in DIRECTIONS and c == "us":
            out.append(DIRECTIONS[t]); continue
        if t in ADDR_WORDS:
            v = ADDR_WORDS[t]
            if v:
                out.extend(v.split())
            continue
        out.append(t)
    return out


NUM_RE = re.compile(r"\d+")
FRACTION_RE = re.compile(r"(?<=\d )[13]/[24]\b")
STANDALONE_NUM_RE = re.compile(r"(?<![a-z0-9])\d+(?![0-9]|st\b|nd\b|rd\b|th\b)")


def _standalone_nums(low: str):
    """House/plot numbers: digit runs that are not ordinals (127th) and not the '1/2' of '890 1/2'."""
    low = FRACTION_RE.sub(" ", low)
    low = re.sub(r"[-/#.,:()]", " ", low)
    return STANDALONE_NUM_RE.findall(low)


def normalize_address(raw: str, country: str) -> dict:
    s = raw or ""
    s = unicodedata.normalize("NFKC", s).replace("°", " ").replace("º", " ")
    s = ZW_RE.sub("", s)
    c = (country or "").lower()
    _, segmap = _translit_maps()
    segs_raw = [p.strip() for p in s.split(",")]
    segs = []
    state = ""
    for p in segs_raw:
        if not p or PLACEHOLDER_RE.match(p):
            continue
        p_clean = p.strip().strip('"').strip()
        if INDIC_RE.search(p_clean):
            if p_clean in segmap:
                state = segmap[p_clean]; continue
            p_clean = translit_text(p_clean)
        segs.append(p_clean)

    street_toks, num, city, locs, all_toks = [], "", "", [], []
    units = []
    nums = []
    for p in segs:
        low = ascii_lower(p)
        low = re.sub(r"\bnull\b", " ", low)
        low_plain = re.sub(r"[^a-z0-9 ]+", " ", low.replace("-", " "))
        low_plain = WS_RE.sub(" ", low_plain).strip()
        if not low_plain:
            continue
        st = _canon_state(low_plain, c)
        if st:
            state = state or st
            continue
        if low_plain.startswith("city of "):
            low_plain = low_plain[8:]
        if UNIT_RE.match(low_plain):
            units.append(low_plain)
            continue
        toks = _canon_addr_tokens(low, c)
        seg_nums = _standalone_nums(low)
        nums.extend(seg_nums)
        has_num = bool(seg_nums)
        is_street = has_num or any(t in STREET_TYPES.values() for t in toks[-2:])
        if is_street and not street_toks and c != "india":
            if seg_nums:
                num = num or seg_nums[0]
            street_toks = [t for t in toks if not NUM_RE.fullmatch(t) and t not in NUM_LABELS
                           and not re.fullmatch(r"\d+[a-z]?|\d+/\d+|[a-z]\d+|[a-z]", t)]
        else:
            place = " ".join(t for t in toks if not re.search(r"\d", t) and t not in NUM_LABELS and len(t) > 1)
            place = re.sub(r"^city of ", "", place)
            place = re.sub(r"\b(city|cdp|twp|township|tq|taluk|tehsil)$", "", place).strip()
            place = CITY_ALIAS.get(place, place)
            if place:
                locs.append(place)
        all_toks.extend(toks)

    if c == "india":
        # India: first number token of the whole address is the closest thing to a house number
        num = nums[0] if nums else ""
    if locs:
        city = locs[-1]
    a_nums = sorted({n.lstrip("0") or "0" for n in nums})
    full = [t for t in all_toks if t not in NUM_LABELS or re.search(r"[a-z]{3}", t)]
    return {
        "a_full": " ".join(full),
        "a_street": " ".join(street_toks),
        "a_num": num,
        "a_nums": " ".join(a_nums),
        "a_city": city,
        "a_locs": "|".join(dict.fromkeys(locs)),
        "a_state": state,
        "a_unit": " ".join(units),
        "a_empty": not all_toks,
    }


# ============================================================== dataframe API
NAME_COLS = ["n_full", "n_core", "n_alt", "n_nospace", "n_skel", "legal", "is_domain", "is_hashtag",
             "has_dba", "was_indic"]
ADDR_COLS = ["a_full", "a_street", "a_num", "a_nums", "a_city", "a_locs", "a_state", "a_unit", "a_empty"]


def normalize_records(names, addresses, countries):
    """Lists in -> dict of column lists out (pure python, picklable for multiprocessing)."""
    cols = {k: [] for k in NAME_COLS + ADDR_COLS}
    for n, a, c in zip(names, addresses, countries):
        d = normalize_name(n)
        d.update(normalize_address(a, c))
        for k in cols:
            cols[k].append(d[k])
    return cols


def _chunk_worker(args):
    return normalize_records(*args)


def normalize_df(df, n_jobs=None, chunk=50_000):
    """df with entity_id, business_name, business_address, country -> normalised DataFrame."""
    import pandas as pd
    from multiprocessing import Pool
    names = df.business_name.tolist(); addrs = df.business_address.tolist(); ctr = df.country.tolist()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk], ctr[i:i + chunk]) for i in range(0, len(names), chunk)]
    n_jobs = n_jobs or os.cpu_count()
    if n_jobs > 1 and len(jobs) > 1:
        with Pool(n_jobs) as pool:
            parts = pool.map(_chunk_worker, jobs)
    else:
        parts = [_chunk_worker(j) for j in jobs]
    out = {"entity_id": df.entity_id.to_numpy(), "country": df.country.to_numpy()}
    for k in NAME_COLS + ADDR_COLS:
        out[k] = [v for p in parts for v in p[k]]
    res = pd.DataFrame(out)
    for k in res.columns:
        if res[k].dtype == object:
            res[k] = res[k].astype("string[pyarrow]")
    return res
