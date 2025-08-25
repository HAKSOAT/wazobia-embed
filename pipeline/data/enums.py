from enum import StrEnum, EnumMeta


class StrEnumMeta(EnumMeta):
    def __repr__(cls):
        values = ", ".join([value for member, value in cls.__members__.items()])
        return "[" + values + "]"

class DataSource(StrEnum, metaclass=StrEnumMeta):
    wura = "wura"
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
