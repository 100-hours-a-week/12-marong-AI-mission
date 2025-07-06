from postprocess.config import EMOJI_MAP

class EmojiGen:
  def __init__(self):
    self.emoji_map = EMOJI_MAP
    
  def add_emojis(self, text):
    for keyword, emoji in self.emoji_map.items():
        if keyword in text:
            text += f" {emoji}"
    return text