import re
from typing import Optional

digit_re = re.compile(r"(\d+[\d\s]*)")

def extract_price_int(price_str: str) -> int:
    if not price_str:
        return 0
    s = price_str.replace("\xa0", " ").replace("\u202f", " ")
    m = digit_re.search(s)
    if not m:
        return 0
    digits = re.sub(r"\s+", "", m.group(1))
    try:
        return int(digits)
    except:
        return 0

def parse_wattage(spec_val: str) -> Optional[int]:
    if not spec_val:
        return None
    m = re.search(r"(\d+)\s*W|(\d+)\s*Вт|(\d+)\s*Watt", spec_val, re.I)
    if m:
        for g in m.groups():
            if g:
                try:
                    return int(g)
                except:
                    pass
    # try search digits
    m2 = re.search(r"(\d{3,4})", spec_val)
    if m2:
        try:
            return int(m2.group(1))
        except:
            pass
    return None

def norm_key(k: str) -> str:
    return re.sub(r"\s+", " ", k.strip().lower())
