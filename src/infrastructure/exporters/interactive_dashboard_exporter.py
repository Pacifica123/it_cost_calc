"""Offline interactive dashboard exporter for DecisionReport.

The dashboard is intentionally dependency-free: it serializes a compact
analytical projection of DecisionReport into a single HTML file with embedded
CSS/JavaScript.  It does not rerun GA/AHP/Pareto/Hybrid and does not alter the
project's decision logic.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

DASHBOARD_SCHEMA_VERSION = 1


def build_interactive_dashboard_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    """Build a compact visualization payload from an existing DecisionReport."""
    candidates = [item for item in _sequence(report.get("candidate_configurations")) if isinstance(item, Mapping)]
    evidence, method_meta = _method_evidence(_mapping(report.get("analysis_results")))
    recommended = _mapping(_mapping(report.get("winner_explanation")).get("recommended"))
    recommended_id = str(recommended.get("id") or "")
    recommended_scope = str(recommended.get("scope") or "")

    candidate_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = str(candidate.get("id") or f"candidate-{index}")
        totals = _mapping(candidate.get("totals"))
        tco = _mapping(totals.get("tco"))
        metadata = _mapping(candidate.get("metadata"))
        metrics = _mapping(candidate.get("metrics"))
        scope = str(candidate.get("scope") or metadata.get("analysis_scope") or "unknown")
        candidate_key = _evidence_key(scope, candidate_id)
        method_rows = {
            **evidence.get(_evidence_key("all", candidate_id), {}),
            **evidence.get(candidate_key, {}),
        }
        is_recommended = candidate_id == recommended_id and (
            not recommended_scope or recommended_scope == scope
        )
        candidate_rows.append(
            {
                "key": candidate_key,
                "id": candidate_id,
                "name": str(candidate.get("name") or candidate_id),
                "scope": scope,
                "source": str(candidate.get("source") or metadata.get("candidate_pool_method") or "—"),
                "candidate_pool_source": str(metadata.get("candidate_pool_source") or "—"),
                "recommended": is_recommended,
                "capex": _first_number(
                    totals.get("capital_cost"),
                    totals.get("capex"),
                    totals.get("initial_investment"),
                ),
                "annual_opex": _first_number(
                    tco.get("annual_opex"),
                    totals.get("annual_opex"),
                    _multiply_optional(tco.get("monthly_opex"), 12.0),
                ),
                "tco": _first_number(
                    tco.get("total_ownership_cost"),
                    totals.get("total_ownership_cost"),
                    totals.get("tco"),
                ),
                "analysis_support": _rank_support(method_rows),
                "method_evidence": deepcopy(method_rows),
                "metrics": _dashboard_metrics(totals, metrics),
                "component_count": len(_sequence(candidate.get("components"))),
            }
        )

    catalog_quality = _mapping(report.get("catalog_data_quality"))
    quality_summary = _mapping(catalog_quality.get("summary"))
    warnings = [str(item) for item in _sequence(report.get("warnings"))]
    risks = _sequence(report.get("risks"))
    project = _mapping(report.get("project"))

    return {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "title": "Интерактивный аналитический дашборд выбора ИТ-решения",
        "project": {
            "title": str(project.get("title") or report.get("title") or "ИТ-решение"),
            "goal": str(project.get("goal") or "—"),
            "generated_at": project.get("generated_at") or _mapping(report.get("metadata")).get("generated_at"),
        },
        "recommended": {
            "id": recommended_id or None,
            "name": recommended.get("name") or recommended.get("id"),
            "method": recommended.get("method"),
            "score": _optional_number(recommended.get("score")),
        },
        "candidates": candidate_rows,
        "methods": method_meta,
        "catalog_quality": {
            "total": _integer(quality_summary.get("catalog_components_total")),
            "complete": _integer(quality_summary.get("complete_metrics")),
            "incomplete": _integer(quality_summary.get("incomplete_metrics")),
            "with_warnings": _integer(quality_summary.get("with_warnings")),
            "with_manual_overrides": _integer(quality_summary.get("with_manual_overrides")),
        },
        "warnings": warnings,
        "risk_count": len(risks),
        "analysis_notes": [
            "Дашборд визуализирует уже рассчитанный DecisionReport и не запускает методы заново.",
            "Индекс аналитической поддержки — диагностический показатель визуализации: среднее нормализованных позиций кандидата в доступных ранжированиях. Он не заменяет GA, AHP, Pareto или Hybrid и не участвует в выборе победителя.",
            "Источник исходных каталожных данных не определяется этим модулем; provenance и предупреждения берутся только из DecisionReport.",
        ],
    }


def build_interactive_dashboard_html(report: Mapping[str, Any]) -> str:
    """Return a standalone UTF-8 HTML dashboard with no external assets."""
    payload = build_interactive_dashboard_payload(report)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Prevent a report string from terminating the script tag.
    serialized = serialized.replace("<", "\\u003c")
    return _HTML_TEMPLATE.replace("__DASHBOARD_DATA__", serialized)


def export_interactive_dashboard(report: Mapping[str, Any], filename: str | Path) -> Path:
    """Write the standalone interactive dashboard to *filename*."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_interactive_dashboard_html(report), encoding="utf-8")
    return path


def _method_evidence(
    analysis_results: Mapping[str, Any],
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    evidence: dict[str, dict[str, dict[str, Any]]] = {}
    method_meta: list[dict[str, Any]] = []

    for key, label in (
        ("genetic_optimization", "GA"),
        ("ahp", "AHP"),
        ("criteria_importance", "Pareto"),
        ("hybrid_assessment", "Hybrid"),
    ):
        raw = analysis_results.get(key)
        for scope, payload in _scoped_payloads(raw):
            ranking = _ranking_rows(key, payload)
            if not ranking:
                continue
            method_id = f"{label}:{scope or 'all'}"
            method_meta.append(
                {
                    "id": method_id,
                    "label": label,
                    "scope": scope or "all",
                    "candidate_count": len(ranking),
                }
            )
            for position, item in enumerate(ranking, start=1):
                candidate_id = str(item.get("id") or item.get("candidate_id") or "")
                if not candidate_id:
                    continue
                rank = _positive_int(item.get("rank"), fallback=position)
                evidence_key = _evidence_key(scope or "all", candidate_id)
                evidence.setdefault(evidence_key, {})[method_id] = {
                    "method": label,
                    "scope": scope or "all",
                    "rank": rank,
                    "candidate_count": len(ranking),
                    "score": _optional_number(
                        item.get("score", item.get("hybrid_score", item.get("ga_score")))
                    ),
                    "pareto_status": item.get("pareto_status"),
                }

    return evidence, method_meta


def _ranking_rows(key: str, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if key == "ahp":
        final = _mapping(payload.get("final"))
        return _normalize_ranking(final.get("ranking"))
    if key == "hybrid_assessment":
        return _normalize_ranking(payload.get("ranking"))
    if key == "criteria_importance":
        ranking = _normalize_ranking(payload.get("ranking"))
        if ranking:
            return ranking
        nondominated = _sequence(payload.get("final_nondominated"))
        return [
            {"id": str(item.get("id") if isinstance(item, Mapping) else item), "rank": index}
            for index, item in enumerate(nondominated, start=1)
        ]
    if key == "genetic_optimization":
        for field in ("candidate_solutions", "candidate_configurations", "ranking"):
            rows = _normalize_ranking(payload.get(field))
            if rows:
                for index, row in enumerate(rows, start=1):
                    metadata = _mapping(row.get("metadata"))
                    metrics = _mapping(row.get("metrics"))
                    row.setdefault("rank", _positive_int(metadata.get("rank"), fallback=index))
                    row.setdefault(
                        "score",
                        metrics.get("ga_score", metrics.get("score", row.get("ga_score"))),
                    )
                return rows
    return []


def _normalize_ranking(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(_sequence(value), start=1):
        if isinstance(item, Mapping):
            row = {str(key): deepcopy(val) for key, val in item.items()}
            row.setdefault("rank", index)
            rows.append(row)
        elif isinstance(item, (list, tuple)) and item:
            rows.append(
                {
                    "id": str(item[0]),
                    "score": item[1] if len(item) > 1 else None,
                    "rank": index,
                }
            )
        elif item not in (None, ""):
            rows.append({"id": str(item), "rank": index})
    return rows


def _scoped_payloads(value: Any) -> list[tuple[str | None, Mapping[str, Any]]]:
    if not isinstance(value, Mapping):
        return []
    by_scope = value.get("by_scope")
    if isinstance(by_scope, Mapping):
        return [
            (str(scope), payload)
            for scope, payload in by_scope.items()
            if isinstance(payload, Mapping)
        ]
    return [(None, value)]


def _rank_support(method_rows: Mapping[str, Mapping[str, Any]]) -> float | None:
    values: list[float] = []
    for row in method_rows.values():
        rank = _positive_int(row.get("rank"), fallback=1)
        count = _positive_int(row.get("candidate_count"), fallback=1)
        if count <= 1:
            values.append(100.0)
        else:
            clipped_rank = max(1, min(rank, count))
            values.append((count - clipped_rank) / (count - 1) * 100.0)
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _dashboard_metrics(totals: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in (
        "total_ram_gb",
        "total_cpu_cores",
        "total_storage_gb",
        "total_max_power_watts",
        "client_seats",
        "license_units",
    ):
        if field in totals and totals[field] not in (None, ""):
            result[field] = deepcopy(totals[field])
    for field in ("ga_score", "score", "reliability_score", "support_score", "functionality_score"):
        if field in metrics and metrics[field] not in (None, ""):
            result[field] = deepcopy(metrics[field])
    return result


def _evidence_key(scope: str, candidate_id: str) -> str:
    return f"{scope}::{candidate_id}"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _optional_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = _optional_number(value)
        if number is not None:
            return number
    return None


def _multiply_optional(value: Any, multiplier: float) -> float | None:
    number = _optional_number(value)
    return None if number is None else number * multiplier


def _integer(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _positive_int(value: Any, *, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return parsed if parsed > 0 else fallback


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Интерактивный аналитический дашборд</title>
<style>
:root{font-family:Inter,Segoe UI,Arial,sans-serif;color:#1f2937;background:#f3f4f6;line-height:1.4}
*{box-sizing:border-box} body{margin:0} .wrap{max-width:1440px;margin:0 auto;padding:24px}
.header,.panel,.card{background:#fff;border:1px solid #e5e7eb;border-radius:16px;box-shadow:0 8px 24px rgba(15,23,42,.05)}
.header{padding:22px 24px;margin-bottom:16px}.header h1{margin:0 0 6px;font-size:28px}.muted{color:#6b7280}
.controls{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}.controls label{font-size:13px;color:#4b5563}.controls select{display:block;margin-top:4px;min-width:220px;padding:9px 10px;border:1px solid #d1d5db;border-radius:10px;background:#fff}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.card{padding:16px}.card .k{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:#6b7280}.card .v{font-size:24px;font-weight:700;margin-top:5px}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}.panel{padding:18px;min-width:0}.panel h2{font-size:18px;margin:0 0 4px}.panel p{margin:0 0 14px;font-size:13px;color:#6b7280}
.full{grid-column:1/-1}.chart{min-height:300px;position:relative}.empty{padding:42px 12px;text-align:center;color:#6b7280;border:1px dashed #d1d5db;border-radius:12px}
svg{width:100%;height:320px;overflow:visible}.axis{stroke:#cbd5e1;stroke-width:1}.gridline{stroke:#e5e7eb;stroke-width:1}.dot{cursor:pointer;transition:.15s}.dot:hover{r:9}.recommended{stroke:#111827;stroke-width:3}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#4b5563;margin-top:8px}.legend i{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:5px}
.rank-row,.cost-row{display:grid;grid-template-columns:minmax(120px,1.2fr) 3fr auto;gap:10px;align-items:center;margin:9px 0}.track{height:20px;border-radius:999px;background:#eef2f7;overflow:hidden}.bar{height:100%;border-radius:999px;background:#64748b}.bar.good{background:#0f766e}.bar.warn{background:#b45309}.bar.accent{background:#2563eb}
.stack{display:flex;height:24px;border-radius:999px;overflow:hidden;background:#eef2f7}.capex{background:#475569}.opex{background:#94a3b8}.small{font-size:12px;color:#6b7280}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:9px;border-bottom:1px solid #e5e7eb;vertical-align:top}th{color:#4b5563;font-weight:600}
.note{padding:12px 14px;border-left:4px solid #64748b;background:#f8fafc;margin:8px 0;font-size:13px}.warning{border-left-color:#b45309;background:#fffbeb}
@media(max-width:980px){.grid{grid-template-columns:repeat(2,1fr)}.panels{grid-template-columns:1fr}.full{grid-column:auto}}
@media(max-width:560px){.wrap{padding:12px}.grid{grid-template-columns:1fr}.header h1{font-size:22px}.rank-row,.cost-row{grid-template-columns:1fr}.controls select{min-width:100%}}
</style>
</head>
<body><div class="wrap">
<section class="header"><h1 id="title"></h1><div id="subtitle" class="muted"></div><div class="controls">
<label>Область анализа<select id="scopeFilter"></select></label><label>Выбранная альтернатива<select id="candidateFilter"></select></label>
</div></section>
<section class="grid" id="kpis"></section>
<section class="panels">
<div class="panel"><h2>Карта альтернатив</h2><p>TCO по оси X; диагностический индекс аналитической поддержки по оси Y. Наведение показывает значения, клик выбирает альтернативу.</p><div class="chart" id="scatter"></div></div>
<div class="panel"><h2>Ранги аналитических методов</h2><p>Позиции выбранной альтернативы в доступных ранжированиях; меньший ранг лучше.</p><div id="ranks"></div></div>
<div class="panel"><h2>Структура затрат</h2><p>Сопоставление CAPEX и годового OPEX по альтернативам в выбранной области.</p><div id="costs"></div></div>
<div class="panel"><h2>Качество исходных данных</h2><p>Сводка provenance/полноты технических метрик из DecisionReport.</p><div id="quality"></div></div>
<div class="panel full"><h2>Сводная таблица</h2><p>Проверяемый табличный слой под визуализациями.</p><div style="overflow:auto"><table><thead><tr><th>Альтернатива</th><th>Область</th><th>TCO</th><th>CAPEX</th><th>OPEX/год</th><th>Поддержка</th><th>Методы</th></tr></thead><tbody id="tableBody"></tbody></table></div></div>
<div class="panel full"><h2>Методические оговорки</h2><div id="notes"></div></div>
</section></div>
<script id="dashboard-data" type="application/json">__DASHBOARD_DATA__</script>
<script>
const D=JSON.parse(document.getElementById('dashboard-data').textContent);const $=id=>document.getElementById(id);
const fmt=n=>n==null?'—':new Intl.NumberFormat('ru-RU',{maximumFractionDigits:2}).format(n);const money=n=>n==null?'—':fmt(n)+' ₽';
const scopeNames={technical:'ТО',software:'ПО',unknown:'Не указано',all:'Все'};let scope='all',selected='';
function visible(){return D.candidates.filter(c=>scope==='all'||c.scope===scope)}
function init(){ $('title').textContent=D.title;$('subtitle').textContent=(D.project.title||'')+' · '+(D.project.goal||'');
 const scopes=['all',...new Set(D.candidates.map(c=>c.scope))];$('scopeFilter').innerHTML=scopes.map(s=>`<option value="${esc(s)}">${esc(scopeNames[s]||s)}</option>`).join('');
 $('scopeFilter').onchange=e=>{scope=e.target.value;selected='';render()};$('candidateFilter').onchange=e=>{selected=e.target.value;renderDetails()}; render(); }
function render(){const rows=visible();if(!selected||!rows.some(c=>c.key===selected)){selected=(rows.find(c=>c.recommended)||rows[0]||{}).key||''}
 $('candidateFilter').innerHTML=rows.map(c=>`<option value="${esc(c.key)}" ${c.key===selected?'selected':''}>${esc(c.name)}</option>`).join('');renderKpis(rows);renderScatter(rows);renderCosts(rows);renderQuality();renderTable(rows);renderDetails();renderNotes();}
function renderKpis(rows){const tcos=rows.map(c=>c.tco).filter(n=>n!=null);const rec=rows.find(c=>c.recommended)||D.candidates.find(c=>c.recommended);const support=rows.map(c=>c.analysis_support).filter(n=>n!=null);
 const cards=[['Альтернатив',rows.length],['Рекомендация',rec?rec.name:'—'],['Диапазон TCO',tcos.length?money(Math.min(...tcos))+' – '+money(Math.max(...tcos)):'—'],['Средняя поддержка',support.length?fmt(support.reduce((a,b)=>a+b,0)/support.length)+'%':'—']];
 $('kpis').innerHTML=cards.map(([k,v])=>`<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(String(v))}</div></div>`).join('')}
function renderScatter(rows){const valid=rows.filter(c=>c.tco!=null&&c.analysis_support!=null);if(!valid.length){$('scatter').innerHTML='<div class="empty">Недостаточно TCO/ранжирований для карты.</div>';return}
 const W=640,H=300,p=45,xv=valid.map(c=>c.tco),yv=valid.map(c=>c.analysis_support),xmin=Math.min(...xv),xmax=Math.max(...xv),ymin=Math.min(...yv,0),ymax=Math.max(...yv,100);const spanX=(xmax-xmin)||1,spanY=(ymax-ymin)||1;
 const x=n=>p+(n-xmin)/spanX*(W-2*p),y=n=>H-p-(n-ymin)/spanY*(H-2*p);let s=`<svg viewBox="0 0 ${W} ${H}" role="img">`;
 for(let i=0;i<=4;i++){const gx=p+i*(W-2*p)/4,gy=p+i*(H-2*p)/4;s+=`<line class="gridline" x1="${gx}" y1="${p}" x2="${gx}" y2="${H-p}"/><line class="gridline" x1="${p}" y1="${gy}" x2="${W-p}" y2="${gy}"/>`}
 s+=`<line class="axis" x1="${p}" y1="${H-p}" x2="${W-p}" y2="${H-p}"/><line class="axis" x1="${p}" y1="${p}" x2="${p}" y2="${H-p}"/>`;
 valid.forEach(c=>{const cls='dot '+(c.recommended?'recommended':'');const fill=c.scope==='technical'?'#2563eb':c.scope==='software'?'#0f766e':'#64748b';s+=`<circle class="${cls}" data-id="${esc(c.key)}" cx="${x(c.tco)}" cy="${y(c.analysis_support)}" r="7" fill="${fill}"><title>${esc(c.name)}\nTCO: ${money(c.tco)}\nПоддержка: ${fmt(c.analysis_support)}%</title></circle>`});s+='</svg><div class="legend"><span><i style="background:#2563eb"></i>ТО</span><span><i style="background:#0f766e"></i>ПО</span><span>Обводка — рекомендованный вариант</span></div>';$('scatter').innerHTML=s;document.querySelectorAll('.dot').forEach(el=>el.onclick=()=>{selected=el.dataset.id;$('candidateFilter').value=selected;renderDetails()})}
function renderDetails(){const c=D.candidates.find(x=>x.key===selected);if(!c){$('ranks').innerHTML='<div class="empty">Нет выбранной альтернативы.</div>';return}const entries=Object.entries(c.method_evidence||{});if(!entries.length){$('ranks').innerHTML='<div class="empty">Для альтернативы нет ранжирований.</div>';return}
 $('ranks').innerHTML=entries.map(([id,r])=>{const width=r.candidate_count<=1?100:Math.max(8,(r.candidate_count-r.rank+1)/r.candidate_count*100);return `<div class="rank-row"><div><b>${esc(r.method)}</b><div class="small">${esc(scopeNames[r.scope]||r.scope)}</div></div><div class="track"><div class="bar accent" style="width:${width}%"></div></div><div>#${r.rank}/${r.candidate_count}</div></div>`}).join('')}
function renderCosts(rows){const max=Math.max(1,...rows.map(c=>(c.capex||0)+(c.annual_opex||0)));$('costs').innerHTML=rows.length?rows.map(c=>{const cap=(c.capex||0)/max*100,op=(c.annual_opex||0)/max*100;return `<div class="cost-row"><div>${esc(c.name)}${c.recommended?' ★':''}<div class="small">TCO ${money(c.tco)}</div></div><div class="stack"><div class="capex" style="width:${cap}%" title="CAPEX ${money(c.capex)}"></div><div class="opex" style="width:${op}%" title="OPEX/год ${money(c.annual_opex)}"></div></div><div class="small">${money((c.capex||0)+(c.annual_opex||0))}</div></div>`}).join(''):'<div class="empty">Нет альтернатив.</div>'}
function renderQuality(){const q=D.catalog_quality,total=Math.max(q.total,1);const rows=[['Полные метрики',q.complete,'good'],['Неполные метрики',q.incomplete,'warn'],['С предупреждениями',q.with_warnings,'warn'],['Ручные уточнения',q.with_manual_overrides,'accent']];$('quality').innerHTML=q.total?rows.map(([k,v,cl])=>`<div class="rank-row"><div>${esc(k)}</div><div class="track"><div class="bar ${cl}" style="width:${v/total*100}%"></div></div><div>${v}/${q.total}</div></div>`).join(''):'<div class="empty">Каталожные данные в отчёте отсутствуют.</div>'}
function renderTable(rows){$('tableBody').innerHTML=rows.map(c=>`<tr><td><b>${esc(c.name)}</b>${c.recommended?' ★':''}<div class="small">${esc(c.id)}</div></td><td>${esc(scopeNames[c.scope]||c.scope)}</td><td>${money(c.tco)}</td><td>${money(c.capex)}</td><td>${money(c.annual_opex)}</td><td>${c.analysis_support==null?'—':fmt(c.analysis_support)+'%'}</td><td>${esc(Object.values(c.method_evidence||{}).map(r=>r.method+' #'+r.rank).join('; ')||'—')}</td></tr>`).join('')}
function renderNotes(){let html=D.analysis_notes.map(n=>`<div class="note">${esc(n)}</div>`).join('');if(D.warnings.length){html+=`<div class="note warning"><b>Предупреждения DecisionReport:</b><br>${D.warnings.slice(0,8).map(esc).join('<br>')}</div>`}$('notes').innerHTML=html}
function esc(v){return String(v??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]))}init();
</script></body></html>'''


__all__ = [
    "DASHBOARD_SCHEMA_VERSION",
    "build_interactive_dashboard_html",
    "build_interactive_dashboard_payload",
    "export_interactive_dashboard",
]
