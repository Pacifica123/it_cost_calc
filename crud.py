# crud.py
class CRUD:
    def __init__(self):
        self.entities = {}

    def add(self, entity_name, data, table):
        if entity_name not in self.entities:
            self.entities[entity_name] = []
        self.entities[entity_name].append(data)
        table.insert("", "end", values=list(data.values()))

    def delete(self, entity_name, index, table):
        del self.entities[entity_name][index]
        for item in table.get_children():
            table.delete(item)
        for row in self.entities[entity_name]:
            table.insert("", "end", values=list(row.values()))

    def update(self, entity_name, index, data, table):
        self.entities[entity_name][index] = data
        for item in table.get_children():
            table.delete(item)
        for row in self.entities[entity_name]:
            table.insert("", "end", values=list(row.values()))
