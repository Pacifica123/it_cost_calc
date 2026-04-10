class CriteriaImportancePresenterMixin:
    def _set_result(self, text):
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)

    def _format_matrix(self, title, matrix, alternatives):
        alt_name = {a["id"]: a["name"] for a in alternatives}
        lines = [title]
        ids = [a["id"] for a in alternatives]
        header = " " * 20 + "".join(f"{alt_name[aid]:>18}" for aid in ids)
        lines.append(header)
        for aid in ids:
            row = [f"{alt_name[aid]:<20}"]
            for bid in ids:
                row.append(f"{matrix[aid][bid]:>18}")
            lines.append("".join(row))
        return "\n".join(lines)

    def _format_report(self, report):
        alternatives = self._alternatives()
        criteria_by_id = {c["id"]: c["name"] for c in self._criteria()}
        lines = [
            f"Кейс: {report['case_name']}",
            "",
            "1. Анализ без учета важности критериев",
            f"Недоминируемые альтернативы: {', '.join(self._alternative_name(aid) for aid in report['raw_nondominated'])}",
            self._format_matrix(
                "Матрица доминирования (базовая):", report["raw_dominance"], alternatives
            ),
            "",
            "2. Построенная N-кратная модель",
        ]

        model = report["analysis_model"]
        for criterion_id in model["criterion_order"]:
            lines.append(
                f"- {criterion_id}: {criteria_by_id[criterion_id]} | кратность {model['criterion_multiplicity'][criterion_id]}"
            )

        lines.extend(
            [
                "",
                "3. Итог после учета важности критериев",
                f"Недоминируемые альтернативы: {', '.join(self._alternative_name(aid) for aid in report['final_nondominated'])}",
                self._format_matrix(
                    "Матрица доминирования (итоговая):", report["final_dominance"], alternatives
                ),
                "",
                "4. Ранжирование",
            ]
        )

        for pos, item in enumerate(report["ranking"], start=1):
            lines.append(
                f"{pos}. {item['name']} | weighted_sum={item['weighted_sum']:.2f} | nondominated={'да' if item['nondominated'] else 'нет'}"
            )

        lines.extend(
            [
                "",
                "5. Пояснение",
                report["explanation"],
            ]
        )
        return "\n".join(lines)
