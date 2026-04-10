import tkinter as tk
from tkinter import messagebox


class CriteriaImportanceTablesMixin:
    def _refresh_scores_tree(self):
        for item in self.scores_tree.get_children():
            self.scores_tree.delete(item)
        if not self.case_data:
            self.scores_tree["columns"] = ()
            return

        alternatives = self._alternatives()
        columns = ["criterion_id", "criterion_name"] + [alt["id"] for alt in alternatives]
        self.scores_tree["columns"] = columns

        headings = {
            "criterion_id": "ID",
            "criterion_name": "Критерий",
        }
        headings.update({alt["id"]: alt["name"] for alt in alternatives})

        for col in columns:
            self.scores_tree.heading(col, text=headings[col])
            width = 90 if col == "criterion_id" else 280 if col == "criterion_name" else 110
            anchor = "w" if col == "criterion_name" else "center"
            self.scores_tree.column(col, width=width, stretch=True, anchor=anchor)

        for criterion in self._criteria():
            values = [criterion["id"], criterion["name"]]
            for alt in alternatives:
                values.append(self.case_data["scores"][alt["id"]].get(criterion["id"], ""))
            self.scores_tree.insert("", "end", iid=criterion["id"], values=values)

    def _refresh_relations_tree(self):
        for item in self.relations_tree.get_children():
            self.relations_tree.delete(item)
        if not self.case_data:
            return
        for idx, rel in enumerate(self.case_data.get("relations", [])):
            self.relations_tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(rel["left"], rel["op"], rel.get("factor", 1), rel["right"]),
            )

    def _clear_result_tables(self):
        for tree in (self.multiplicity_tree, self.ranking_tree):
            for item in tree.get_children():
                tree.delete(item)

    def _populate_result_tables(self, report):
        self._clear_result_tables()
        criteria_name = {c["id"]: c["name"] for c in self._criteria()}
        model = report["analysis_model"]
        for criterion_id in model["criterion_order"]:
            self.multiplicity_tree.insert(
                "",
                "end",
                values=(
                    criterion_id,
                    criteria_name.get(criterion_id, criterion_id),
                    model["criterion_multiplicity"][criterion_id],
                ),
            )
        for pos, item in enumerate(report["ranking"], start=1):
            self.ranking_tree.insert(
                "",
                "end",
                values=(
                    pos,
                    item["name"],
                    f"{item['weighted_sum']:.2f}",
                    "Да" if item["nondominated"] else "Нет",
                ),
            )

    def _on_score_double_click(self, event):
        if not self.case_data:
            return
        region = self.scores_tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        row_id = self.scores_tree.identify_row(event.y)
        column = self.scores_tree.identify_column(event.x)
        if not row_id or column in ("#1", "#2"):
            return

        col_index = int(column.replace("#", "")) - 1
        columns = list(self.scores_tree["columns"])
        alt_id = columns[col_index]
        current_value = self.case_data["scores"][alt_id].get(row_id, "")

        popup = tk.Toplevel(self)
        popup.title("Редактирование оценки")
        popup.transient(self.winfo_toplevel())
        popup.grab_set()

        tk.Label(
            popup,
            text=f"Критерий {row_id} ({self._criterion_name(row_id)}), система {self._alternative_name(alt_id)}",
        ).pack(padx=10, pady=(10, 4))
        entry = tk.Entry(popup)
        entry.pack(padx=10, pady=4)
        entry.insert(0, str(current_value))
        entry.select_range(0, "end")
        entry.focus_set()

        def save():
            try:
                self.case_data["scores"][alt_id][row_id] = float(entry.get().replace(",", "."))
                self._refresh_scores_tree()
                popup.destroy()
            except Exception:
                messagebox.showerror("Ошибка", "Введите число")

        tk.Button(popup, text="Сохранить", command=save).pack(padx=10, pady=(4, 10))

    def _selected_criterion_id(self):
        selected = self.scores_tree.selection()
        return selected[0] if selected else None
