import shutil

class logposter:
    def __init__(self):
        self.width = self.get_terminal_width()
        self.flex = 6

    def log_box(self, texts, flex=None, style="-"):
        if flex is None:
            flex = self.flex
        if isinstance(texts, list):
            self.max_len = max(len(text) for text in texts)
            self.box_width = self.max_len + flex
        if isinstance(texts, str):
            self.max_len = len(texts)
        
        self.PrintFlexLine(texts, flex, style)
        self.PrintBoxLines(texts, flex, style)
        self.PrintFlexLine(texts, flex, style)


    
    def PrintFlexLine(self, texts, flex=None, style="-"):
        if flex is None:
            flex = self.flex
        length = self.box_width + flex - 4
        print(style * length)

    def PrintBoxLines(self, texts, flex=None, style="-"):
        if flex is None:
                flex = self.flex
        if style == "-":
            style = "|"
        if isinstance(texts, str):
            sp = int(flex/2 - 1)
            print(style + " " * sp + texts + " " * sp + sym)
        elif isinstance(texts, list):
            for text in texts:
                left_padding = (self.box_width - len(text)) // 2
                right_padding = self.box_width - left_padding - len(text)
                print(style + " " * left_padding + text + " " * right_padding + style)


    def get_terminal_width(self):
        terminal_size = shutil.get_terminal_size()
        return terminal_size.columns

lp = logposter()
texts = ["test1", "test24685488888888gyukgfuygiyugyugygyu8", "test3"]
lp.log_box(texts,style="#")
    






