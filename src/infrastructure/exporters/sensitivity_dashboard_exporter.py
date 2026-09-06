"""Standalone exports for the PUAZ decision-sensitivity scenario model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from application.services.decision_sensitivity_analysis_service import (
    DecisionSensitivityAnalysisService,
)


def build_decision_sensitivity_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    """Build the machine-readable sensitivity payload."""
    return DecisionSensitivityAnalysisService().build(report)


def build_decision_sensitivity_html(report: Mapping[str, Any]) -> str:
    """Return a standalone UTF-8 interactive HTML report."""
    payload = build_decision_sensitivity_payload(report)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    return _HTML_TEMPLATE.replace("__SENSITIVITY_DATA__", serialized)


def export_decision_sensitivity_json(report: Mapping[str, Any], filename: str | Path) -> Path:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_decision_sensitivity_payload(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def export_decision_sensitivity_dashboard(
    report: Mapping[str, Any], filename: str | Path
) -> Path:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_decision_sensitivity_html(report), encoding="utf-8")
    return path


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ПУАЗ — анализ чувствительности решения</title>
<style>
:root{color-scheme:dark;--bg:#11151b;--panel:#181e26;--panel2:#202833;--text:#eef2f6;--muted:#aeb8c5;--line:#34404e;--accent:#8bb8ff;--good:#8ed6a0;--warn:#f3cc7a;--bad:#ed9898}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1480px;margin:auto;padding:28px}.eyebrow{color:var(--accent);font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:12px}
h1{font-size:32px;margin:6px 0 8px}p{margin:0}.lead{max-width:980px;color:var(--muted);font-size:16px}.grid{display:grid;gap:16px}.controls{grid-template-columns:repeat(3,minmax(0,1fr));margin:24px 0 16px}
.panel,.card{background:var(--panel);border:1px solid var(--line);border-radius:16px}.panel{padding:18px}.card{padding:16px}.label{display:block;color:var(--muted);font-size:12px;margin-bottom:7px}
select,input[type=range]{width:100%}select{background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:10px;padding:10px}
input[type=range]{accent-color:var(--accent)}.rangeRow{display:flex;justify-content:space-between;gap:10px;color:var(--muted);font-size:12px}.kpis{grid-template-columns:repeat(5,minmax(0,1fr));margin-bottom:16px}.kpi b{display:block;font-size:22px;margin-top:4px}.kpi small{color:var(--muted)}
.two{grid-template-columns:minmax(0,1.25fr) minmax(0,.75fr);margin-bottom:16px}.sectionTitle{font-weight:750;font-size:18px;margin:0 0 12px}.chartBox{min-height:330px;overflow:auto}.chartBox svg{display:block;width:100%;min-width:620px;height:320px}
.heatmap{display:grid;gap:2px;align-items:stretch;min-width:720px}.heatCell{min-height:30px;border-radius:5px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#111;cursor:default}.heatLabel{display:flex;align-items:center;color:var(--muted);font-size:11px;padding:0 4px}.legend{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}.legend span{display:inline-flex;align-items:center;gap:6px;color:var(--muted);font-size:12px}.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
table{width:100%;border-collapse:collapse}th,td{padding:9px 8px;border-bottom:1px solid var(--line);text-align:left}th{font-size:11px;text-transform:uppercase;color:var(--muted)}td.num{text-align:right;font-variant-numeric:tabular-nums}.winner{color:var(--good);font-weight:750}.muted{color:var(--muted)}
.barRow{display:grid;grid-template-columns:minmax(110px,1fr) 2.4fr 56px;gap:8px;align-items:center;margin:9px 0}.bar{height:11px;background:var(--panel2);border-radius:999px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent)}.notes{margin-top:16px}.notes ul{margin:8px 0 0 18px;color:var(--muted)}.warning{color:var(--warn)}.empty{padding:28px;color:var(--muted);text-align:center}
@media(max-width:980px){main{padding:18px}.controls,.kpis,.two{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
</head>
<body><main>
<div class="eyebrow">ПУАЗ · сценарная визуализация</div>
<h1>Анализ чувствительности итоговой рекомендации</h1>
<p class="lead">Модуль исследует, насколько меняется Hybrid-рекомендация при изменении баланса между GA и AHP. TCO-порог дополнительно показывает, что происходит при ограничении уже экспортированного пула альтернатив.</p>

<section class="grid controls">
<div class="panel"><label class="label" for="scopeSelect">Область анализа</label><select id="scopeSelect"></select></div>
<div class="panel"><label class="label" for="lambdaRange">λ — вклад GA в Hybrid</label><input id="lambdaRange" type="range" min="0" max="1" step="0.01" value="0.5"><div class="rangeRow"><span>AHP ← 0</span><b id="lambdaValue">0.50</b><span>1 → GA</span></div></div>
<div class="panel"><label class="label" for="budgetSelect">TCO-порог сценария</label><select id="budgetSelect"></select><div class="rangeRow"><span>Фильтр по экспортированному пулу</span><b id="budgetValue">без ограничения</b></div></div>
</section>

<section id="content">
<div class="grid kpis">
<div class="card kpi"><small>Победитель сценария</small><b id="winnerKpi">—</b></div>
<div class="card kpi"><small>Базовый Hybrid</small><b id="baselineKpi">—</b></div>
<div class="card kpi"><small>Устойчивость базового лидера</small><b id="stabilityKpi">—</b></div>
<div class="card kpi"><small>Смен лидера по λ</small><b id="switchKpi">—</b></div>
<div class="card kpi"><small>Допустимых альтернатив</small><b id="feasibleKpi">—</b></div>
</div>

<div class="grid two">
<div class="panel"><h2 class="sectionTitle">Как меняется Hybrid-score по λ</h2><div id="lineChart" class="chartBox"></div><div id="lineLegend" class="legend"></div></div>
<div class="panel"><h2 class="sectionTitle">Доля лидерства по λ</h2><div id="shareBars"></div><div id="intervals" class="muted"></div></div>
</div>

<div class="panel" style="margin-bottom:16px"><h2 class="sectionTitle">Карта сценариев: TCO-порог × λ</h2><div id="heatmap" class="chartBox"></div><div id="heatLegend" class="legend"></div></div>
<div class="panel"><h2 class="sectionTitle">Рейтинг текущего сценария</h2><div style="overflow:auto"><table><thead><tr><th>#</th><th>Альтернатива</th><th>Hybrid</th><th>GA norm</th><th>AHP norm</th><th>TCO</th><th>Pareto</th></tr></thead><tbody id="rankingBody"></tbody></table></div></div>
</section>

<section class="panel notes"><h2 class="sectionTitle">Методические границы</h2><ul id="notesList"></ul></section>
</main>
<script id="sensitivityData" type="application/json">__SENSITIVITY_DATA__</script>
<script>
const DATA=JSON.parse(document.getElementById('sensitivityData').textContent);
const scopeSelect=document.getElementById('scopeSelect'), lambdaRange=document.getElementById('lambdaRange'), budgetSelect=document.getElementById('budgetSelect');
const palette=['#8bb8ff','#8ed6a0','#f3cc7a','#dba7ff','#ff9f9f','#8fe0d0','#ffb87a','#b9c5ff','#d8df8b','#d7a4a4'];
const fmt=n=>n==null?'—':new Intl.NumberFormat('ru-RU',{maximumFractionDigits:2}).format(n);
const money=n=>n==null?'—':new Intl.NumberFormat('ru-RU',{maximumFractionDigits:0}).format(n);
const clamp=v=>Math.max(0,Math.min(1,Number(v)||0));
function norm(values){const finite=values.filter(v=>Number.isFinite(v));if(!finite.length)return values.map(_=>.5);const lo=Math.min(...finite),hi=Math.max(...finite);if(values.length===1&&finite.length===1)return [1];if(Math.abs(hi-lo)<1e-12)return values.map(_=>.5);return values.map(v=>((Number.isFinite(v)?v:lo)-lo)/(hi-lo));}
function rankScenario(scope,lambda,budget){let rows=scope.candidates.filter(c=>budget===''||budget==='none'||(Number.isFinite(c.tco)&&c.tco<=Number(budget)+1e-9));if(!rows.length)return {ranking:[],winner_id:null,winner_name:null,feasible_count:0};const ga=norm(rows.map(r=>Number.isFinite(r.ga_score)?r.ga_score:null)),ahp=norm(rows.map(r=>Number.isFinite(r.ahp_score)?r.ahp_score:null));rows=rows.map((r,i)=>({...r,ga_score_normalized:ga[i],ahp_score_normalized:ahp[i],hybrid_score:lambda*ga[i]+(1-lambda)*ahp[i],rank_disagreement:Math.abs((r.ga_rank||i+1)-(r.ahp_rank||i+1))}));rows.sort((a,b)=>(b.hybrid_score-a.hybrid_score)||((b.pareto_status==='недоминируемая')-(a.pareto_status==='недоминируемая'))||(a.rank_disagreement-b.rank_disagreement));rows.forEach((r,i)=>r.rank=i+1);return {ranking:rows,winner_id:rows[0]?.id,winner_name:rows[0]?.name,feasible_count:rows.length};}
function colorMap(scope){const ids=[...new Set(scope.candidates.map(c=>c.id))];return Object.fromEntries(ids.map((id,i)=>[id,palette[i%palette.length]]));}
function currentScope(){return DATA.scopes[scopeSelect.value];}
function setup(){const entries=Object.entries(DATA.scopes);scopeSelect.innerHTML=entries.map(([id,s])=>`<option value="${id}">${id==='technical'?'ТО':id==='software'?'ПО':id}</option>`).join('');if(!entries.length){document.getElementById('content').innerHTML='<div class="panel empty">Hybrid-результаты отсутствуют. Сначала выполните расчёты GA/AHP/Hybrid и повторите экспорт.</div>';renderNotes();return;}scopeSelect.addEventListener('change',()=>{syncBudget();render();});lambdaRange.addEventListener('input',render);budgetSelect.addEventListener('change',render);syncBudget();render();renderNotes();}
function syncBudget(){const s=currentScope();const levels=s?.budget_levels||[null];budgetSelect.innerHTML=levels.map(v=>`<option value="${v==null?'none':v}">${v==null?'Без ограничения':money(v)}</option>`).join('');budgetSelect.value='none';lambdaRange.value=String(s?.baseline?.lambda??0.5);}
function render(){const s=currentScope();if(!s)return;const lambda=clamp(lambdaRange.value),budget=budgetSelect.value;const scenario=rankScenario(s,lambda,budget);document.getElementById('lambdaValue').textContent=lambda.toFixed(2);document.getElementById('budgetValue').textContent=budget==='none'?'без ограничения':money(Number(budget));document.getElementById('winnerKpi').textContent=scenario.winner_name||'нет допустимых';document.getElementById('baselineKpi').textContent=s.baseline?.winner_name||'—';document.getElementById('stabilityKpi').textContent=`${fmt(s.stability?.baseline_winner_stability_pct||0)}%`;document.getElementById('switchKpi').textContent=String(s.stability?.switch_count??0);document.getElementById('feasibleKpi').textContent=`${scenario.feasible_count}/${s.candidates.length}`;renderRanking(scenario,s);renderLine(s,lambda);renderShares(s);renderHeatmap(s);}
function renderRanking(scenario,s){const colors=colorMap(s);document.getElementById('rankingBody').innerHTML=scenario.ranking.map(r=>`<tr><td>${r.rank}</td><td class="${r.rank===1?'winner':''}"><span class="dot" style="background:${colors[r.id]}"></span> ${r.name}</td><td class="num">${r.hybrid_score.toFixed(4)}</td><td class="num">${r.ga_score_normalized.toFixed(3)}</td><td class="num">${r.ahp_score_normalized.toFixed(3)}</td><td class="num">${money(r.tco)}</td><td>${r.pareto_status||'—'}</td></tr>`).join('');}
function renderLine(s,currentLambda){const colors=colorMap(s),W=760,H=300,pad={l:42,r:16,t:18,b:34};const xs=x=>pad.l+x*(W-pad.l-pad.r),ys=y=>H-pad.b-y*(H-pad.t-pad.b);const series=Object.fromEntries(s.candidates.map(c=>[c.id,[]]));for(const sweep of s.lambda_sweep||[]){for(const row of sweep.ranking||[]){series[row.id]?.push([sweep.lambda,row.hybrid_score]);}}let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Hybrid-score по lambda"><line x1="${pad.l}" y1="${H-pad.b}" x2="${W-pad.r}" y2="${H-pad.b}" stroke="#566272"/><line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${H-pad.b}" stroke="#566272"/>`;for(let t=0;t<=1.001;t+=.25){svg+=`<line x1="${xs(t)}" y1="${pad.t}" x2="${xs(t)}" y2="${H-pad.b}" stroke="#26313e"/><text x="${xs(t)}" y="${H-10}" fill="#aeb8c5" text-anchor="middle" font-size="11">${t.toFixed(2)}</text>`;}for(let t=0;t<=1.001;t+=.25){svg+=`<line x1="${pad.l}" y1="${ys(t)}" x2="${W-pad.r}" y2="${ys(t)}" stroke="#26313e"/><text x="${pad.l-8}" y="${ys(t)+4}" fill="#aeb8c5" text-anchor="end" font-size="11">${t.toFixed(2)}</text>`;}for(const c of s.candidates){const pts=(series[c.id]||[]).map(([x,y])=>`${xs(x)},${ys(y)}`).join(' ');svg+=`<polyline points="${pts}" fill="none" stroke="${colors[c.id]}" stroke-width="2.4"/>`;}svg+=`<line x1="${xs(currentLambda)}" y1="${pad.t}" x2="${xs(currentLambda)}" y2="${H-pad.b}" stroke="#eef2f6" stroke-dasharray="5 5"/>`;svg+='</svg>';document.getElementById('lineChart').innerHTML=svg;document.getElementById('lineLegend').innerHTML=s.candidates.map(c=>`<span><i class="dot" style="background:${colors[c.id]}"></i>${c.name}</span>`).join('');}
function renderShares(s){const colors=colorMap(s),shares=s.stability?.winner_shares||[];document.getElementById('shareBars').innerHTML=shares.map(x=>`<div class="barRow"><span>${x.name}</span><div class="bar"><i style="width:${x.share_pct}%;background:${colors[x.id]}"></i></div><b>${x.share_pct}%</b></div>`).join('')||'<p class="muted">Нет данных.</p>';const intervals=s.stability?.winner_intervals||[];document.getElementById('intervals').innerHTML=intervals.length?'<p style="margin-top:16px"><b>Интервалы лидерства:</b></p>'+intervals.map(x=>`<div>λ ${Number(x.lambda_from).toFixed(2)}–${Number(x.lambda_to).toFixed(2)} → ${x.winner_name||'нет победителя'}</div>`).join(''):'';}
function renderHeatmap(s){const lambdas=s.lambda_values||[],budgets=s.budget_levels||[null],colors=colorMap(s),map=new Map((s.grid||[]).map(c=>[`${c.tco_limit==null?'none':c.tco_limit}|${c.lambda}`,c]));const cols=lambdas.length+1;let html=`<div class="heatmap" style="grid-template-columns:110px repeat(${lambdas.length},minmax(28px,1fr))"><div></div>${lambdas.map(l=>`<div class="heatLabel" title="lambda=${l}">${Number(l).toFixed(2)}</div>`).join('')}`;for(const b of budgets){html+=`<div class="heatLabel">${b==null?'без лимита':money(b)}</div>`;for(const l of lambdas){const cell=map.get(`${b==null?'none':b}|${l}`)||{};const col=cell.winner_id?colors[cell.winner_id]:'#3a4654';html+=`<div class="heatCell" style="background:${col}" title="λ=${Number(l).toFixed(2)}; TCO=${b==null?'без лимита':money(b)}; победитель=${cell.winner_name||'нет'}; допустимо=${cell.feasible_count||0}">${cell.winner_id||'—'}</div>`;}}html+='</div>';document.getElementById('heatmap').innerHTML=html;document.getElementById('heatLegend').innerHTML=s.candidates.map(c=>`<span><i class="dot" style="background:${colors[c.id]}"></i>${c.id} — ${c.name}</span>`).join('');}
function renderNotes(){const notes=[...(DATA.limitations||[]),...(DATA.warnings||[])];for(const s of Object.values(DATA.scopes||{}))for(const w of s.warnings||[])notes.push(w);document.getElementById('notesList').innerHTML=notes.map(x=>`<li>${x}</li>`).join('');}
setup();
</script></body></html>'''


__all__ = [
    "build_decision_sensitivity_payload",
    "build_decision_sensitivity_html",
    "export_decision_sensitivity_json",
    "export_decision_sensitivity_dashboard",
]
