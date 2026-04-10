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
            # clear current
            self.configurations.clear()
            for item in self.cfg_table.get_children():
                self.cfg_table.delete(item)
            # load
            for c in cfgs:
                cid = c.get("id") or c.get("name") or f"cfg_{len(self.configurations)+1}"
                devices = c.get("devices", [])
                people = int(c.get("meta", {}).get("people", c.get("people", 0)))
                self.configurations[cid] = {"devices": devices, "meta": {"people": people}}
                self.cfg_table.insert("", "end", iid=cid, values=(cid, people, len(devices)))
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить JSON: {e}")

    def _save_configs_to_json(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")]
        )
        if not path:
            return
        out = {"configurations": []}
        for cid, v in self.configurations.items():
            out["configurations"].append(
                {"id": cid, "devices": v["devices"], "meta": v.get("meta", {})}
            )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        messagebox.showinfo("OK", "Сохранено")
