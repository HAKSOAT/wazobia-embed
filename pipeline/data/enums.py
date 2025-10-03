from enum import StrEnum, EnumMeta


class StrEnumMeta(EnumMeta):
    def __repr__(cls):
        values = ", ".join([value for member, value in cls.__members__.items()])
        return "[" + values + "]"

class DataSource(StrEnum, metaclass=StrEnumMeta):
    wura = "wura"
    # The Mato datasource is made up of the following sources:
    # Yoruba -> Aláròyé, VON Yoruba and BBC Yoruba.
    # Igbo -> VON Igbo, and BBC Igbo. 
    # Hausa -> Premium Times Hausa, Fim Magazine, VOA Hausa, Katsina Post, Legit Hausa, Amaniya, and VON Hausa.
    mato = "mato"
    masakhanews = "masakhanews"
    
    def __repr__(self):
        return self.value

class Language(StrEnum, metaclass=StrEnumMeta):
    yoruba = "yoruba"
    igbo = "igbo"
    hausa = "hausa"
    english = "english"
    
    def __repr__(self):
        return self.value

class DataSplit(StrEnum, metaclass=StrEnumMeta):
    train = "train"
    eval = "eval"
    test = "test"
    
    def __repr__(self):
        return self.value

class DataOperation(StrEnum, metaclass=StrEnumMeta):
    create = "create"
    postprocess = "postprocess"
    translate = "translate"

    def __repr__(self):
        return self.value
