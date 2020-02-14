import argparse

class PrettyFormatter(argparse.ArgumentDefaultsHelpFormatter):
   '''For both printing the defaults and new line characters'''

   def _split_lines(self, text, width):
        result = []
        text_lines = text.splitlines()
        for line in text_lines:
           splitted_line = super()._split_lines(line, width)
           if '*' in line:
              pad = '  '
              for i in range(1, len(splitted_line)):
                 splitted_line[i] = pad + splitted_line[i]
           result.extend(splitted_line)

        result+=['']
        return result

