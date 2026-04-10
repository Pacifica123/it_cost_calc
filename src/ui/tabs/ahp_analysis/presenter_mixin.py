import json


class AHPPresenterMixin:
    def _display_summary(self, report):
        self.summary_text.delete("1.0", "end")
        lines = []
        lines.append(f"Всего входных конфигураций: {report.get('total_input')}")
        lines.append(
            f"Прошло фильтр: {report.get('passed_count')} (убрано: {report.get('removed_count')})"
        )
        if "warning" in report:
            lines.append("WARNING: " + report["warning"])
        crit = report.get("criteria", {})
        if crit:
            weights = crit.get("weights", [])
            names = crit.get("names", [])
            lines.append("Критерии (веса):")
            for criterion_id, weight in zip(names, weights):
                criterion_label = self._format_soft_criterion_label(criterion_id)
                lines.append(f"  {criterion_label}: {weight:.4f}")
            lines.append(f"CR(criteria): {crit.get('CR'):.4f}  CI: {crit.get('CI'):.6f}")
        final = report.get("final", {})
        if final:
            lines.append("Рейтинг (top -> ...):")
            for aid, score in final.get("ranking", []):
                config_name = self._configuration_display_name(aid)
                lines.append(f"  {config_name}: {score:.4f}")
            selected = [self._configuration_display_name(item) for item in report.get("selection", {}).get("selected_ids", [])]
            lines.append("Выбранные (top): " + ", ".join(selected))
        self.summary_text.insert("1.0", "\n".join(lines))

    def _fill_details(self, report):
        self.details_panel.delete("1.0", "end")
        pretty = json.dumps(report, ensure_ascii=False, indent=2)
        self.details_panel.insert("1.0", pretty)

    def _toggle_details(self):
        if self.details_shown:
            self.details_panel.pack_forget()
            self.details_shown = False
            self.details_btn.configure(text="Показать детали (CI/CR, матрицы, веса)")
        else:
            self.details_panel.pack(fill="both", expand=True)
            self.details_shown = True
            self.details_btn.configure(text="Скрыть детали")
