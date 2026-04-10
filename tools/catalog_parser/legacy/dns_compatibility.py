import re
from typing import Dict, Optional

from .dns_spec_utils import parse_wattage

def cpu_socket_from_specs(specs: Dict[str, str]) -> Optional[str]:
    for key in specs:
        if "сокет" in key or "socket" in key:
            return specs[key].split(",")[0].strip()
    return None

def mb_socket_from_specs(specs: Dict[str, str]) -> Optional[str]:
    return cpu_socket_from_specs(specs)

def ram_type_from_specs(specs: Dict[str, str]) -> Optional[str]:
    for key in specs:
        if "тип" in key and ("опер" in key or "пам" in key):
            v = specs[key]
            # ищем DDR4/DDR5
            m = re.search(r"(DDR[345])", v, re.I)
            if m:
                return m.group(1).upper()
    # альтернативы
    for key in specs:
        if "ddr" in specs[key].lower():
            m = re.search(r"(DDR[345])", specs[key], re.I)
            if m:
                return m.group(1).upper()
    return None

def formfactor_from_specs(specs: Dict[str, str]) -> Optional[str]:
    for key in specs:
        if "форм" in key or "form-factor" in key.lower():
            v = specs[key].upper()
            if "ATX" in v:
                return "ATX"
            if "MICRO" in v or "mATX" in v or "MICRO ATX" in v:
                return "mATX"
            if "MINI" in v or "ITX" in v:
                return "Mini-ITX"
    return None

def tdp_from_specs(specs: Dict[str, str]) -> Optional[int]:
    for key in specs:
        if "tdp" in key.lower() or "тепловыдел" in key or "теплов" in key:
            m = re.search(r"(\d{2,4})", specs[key])
            if m:
                return int(m.group(1))
    return None

def compatible_pair(cpu: Dict, mb: Dict, ram: Dict, psu: Dict, case: Dict) -> bool:
    # CPU socket vs MB socket
    cs = cpu_socket_from_specs(cpu.get("specs", {}))
    ms = mb_socket_from_specs(mb.get("specs", {}))
    if cs and ms and cs != ms:
        return False
    # RAM type
    rtype = ram_type_from_specs(ram.get("specs", {}))
    mb_rtype = ram_type_from_specs(mb.get("specs", {}))
    if rtype and mb_rtype and rtype != mb_rtype:
        return False
    # case formfactor vs mb formfactor
    mb_ff = formfactor_from_specs(mb.get("specs", {}))
    case_ff = formfactor_from_specs(case.get("specs", {}))
    if mb_ff and case_ff and mb_ff != case_ff:
        # allow case more roomy: ATX case supports mATX, Mini-ITX
        if case_ff == "ATX":
            pass
        else:
            return False
    # PSU wattage vs rough requirement
    cpu_tdp = tdp_from_specs(cpu.get("specs", {})) or 65
    gpu_tdp = tdp_from_specs({}) or 150  # unknown GPU - assume 150
    needed = cpu_tdp + gpu_tdp + 100  # extras
    psu_w = parse_wattage(
        psu.get("specs", {}).get("Мощность", "")
        or psu.get("specs", {}).get("Мощность блока", "")
        or psu.get("price", "")
    )
    if psu_w and psu_w < int(needed * 1.2):
        return False
    return True
