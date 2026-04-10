import json
from tkinter import filedialog, messagebox


class AHPIOMixin:
    def _load_from_json(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfgs = data.get("configurations", data if isinstance(data, list) else [])
            self._replace_configurations(cfgs)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить JSON: {e}")

    def _save_configs_to_json(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")]
        )
        if not path:
            return
        out = {"configurations": []}
        for cid, value in self.configurations.items():
            out["configurations"].append(
                {
                    "id": cid,
                    "name": value.get("name", cid),
                    "devices": value["devices"],
                    "meta": value.get("meta", {}),
                }
            )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        messagebox.showinfo("OK", "Сохранено")
