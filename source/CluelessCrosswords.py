"""Creates a PDF with Clueless Crosswords."""
import os
import sys
import subprocess
import argparse
import tkinter as tk
from tkinter import font
import copy
import gzip
import pickle
from pathlib import Path
from random import shuffle, randrange, choice
from string import ascii_uppercase
import reportlab.lib.styles as styles
from reportlab.lib import colors, enums
from reportlab.lib.pagesizes import letter, A4, inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.platypus.flowables import Flowable
if os.name == "nt":
    import ctypes.wintypes
# import cProfile


class makePuzzles:
    """
    Makes the puzzles in a format ready to be encoded.

    Parameters
    ----------
    numberOfPuzzles : int
        The number of puzzles to create, minimum of 1.
    output : list
        Holds the created puzzles.

    """

    def __init__(self, numberOfPuzzles, output):
        """Process the words used to create the puzzle and make the puzzles."""
        self.ROWS = self.COLUMNS = 13
        self.nPuzzles = numberOfPuzzles
        self.output = output

        assert self.nPuzzles > 0, f"Number of puzzles: {self.nPuzzles} <= 0"
        assert isinstance(self.output, list)

        inputPath = os.path.dirname(Path(sys.argv[0]).absolute())
        inputWords = os.path.join(inputPath, "words.pkl.gz")
        inputUsefull = os.path.join(inputPath, "useful.pkl.gz")

        with gzip.open(inputWords, "rb") as fh:
            self.allWords = pickle.load(fh)
        with gzip.open(inputUsefull, "rb") as fh:
            self.usefullWords = pickle.load(fh)
        self.allWordsSet = set(self.allWords)
        self.allWordsSet.update(ascii_uppercase)
        self.word3 = tuple(sorted([w for w in self.allWords if len(w) == 3]))
        self.word2 = tuple(sorted([w for w in self.allWords if len(w) == 2]))

    def make(self):
        """
        Primary process to create the puzzles.

        Returns
        -------
        shortWordPage : tuple
            A Tuple containing the short words for optional printing.

        """
        for n in range(self.nPuzzles):
            self._addPuzzle()

        for i, line in enumerate(self.output):
            self.output[i] = ''.join(line)
        return (("Two Letter Words", ), self.word2,
                ("Three Letter Words", ), self.word3)

    def _addPuzzle(self):
        """Add a single puzzle to the list of puzzles."""
        puzzle = [[0]*self.COLUMNS for _ in range(self.ROWS)]

        unusedLetters = set(ascii_uppercase)
        usedWordSet = set()

        # Add random word to top & bottom then left & right
        # Makes the puzzle look a bit nicer when printed
        for _ in range(2):
            for row in (0, self.ROWS - 1):
                word = self._getWord(usedWordSet, self.allWords, unusedLetters)
                usedWordSet.add(word)
                startCol = 0 if puzzle[row][0] == 0 else 1
                if puzzle[row][self.COLUMNS - 1] == 0:
                    endCol = self.COLUMNS
                else:
                    endCol = self.COLUMNS - 1
                endCol -= len(word)
                startCol += randrange(startCol, endCol)
                for i in range(len(word)):
                    puzzle[row][startCol + i] = word[i]
            puzzle = self._flipTbl(puzzle)

        loops = 32
        while unusedLetters and loops:
            # Unused letters used to soft limit strange words while using
            # as many letters from the alphabet as reasonably possible.
            loops -= 1
            word = self._getWord(usedWordSet, self.usefullWords, unusedLetters)
            added = self._addWordToPuzzle(word, puzzle)
            if added:
                usedWordSet.add(word)
                unusedLetters.difference_update(word)

        for wl in (self.allWords, self.word3, self.word2):
            loops = 64
            while loops:
                loops -= 1
                word = self._getWord(usedWordSet, wl)
                added = self._addWordToPuzzle(word, puzzle)
                if added:
                    usedWordSet.add(word)
                    unusedLetters.difference_update(word)

        if (len(unusedLetters) > 2 or
                sum([line.count(0) for line in puzzle]) > 90):
            self._addPuzzle()
        else:
            self._fillTbl(puzzle)
            self._verifyWords(puzzle)
            self._cleanPuzzle(puzzle)
            self.output.extend(puzzle)

    def _fillTbl(self, tbl):
        """Add words in a manual fashion rather than fully random."""
        # Other manual sections can be added, but note that this is verbose
        # and error prone. There is a verification at the end to ensure the
        # puzzles still work though. This is potentially significantly faster
        # than the random function however.
        for _ in range(2):
            for row in range(self.ROWS):
                # Check for .L.L. || xL.L. || .L.Lx
                for col in range(self.COLUMNS - 2):
                    words = []
                    if (
                        (tbl[row][col] != 0) and
                        (col == 0 or tbl[row][col-1] == 0) and
                        (tbl[row][col+1] == 0) and (tbl[row][col+2] != 0) and
                        (col+3 == self.COLUMNS or tbl[row][col+3] == 0)
                    ):
                        mCol = col+1
                        if (
                            (row == 0 or tbl[row-1][mCol] == 0) and
                            (row+1 == self.ROWS or tbl[row+1][mCol] == 0)
                        ):
                            start = tbl[row][col]
                            end = tbl[row][col+2]
                            for word in self.word3:
                                if word[0] == start and word[2] == end:
                                    words.append(word)
                                elif word[0] > start:
                                    break
                            if words:
                                shuffle(words)
                                tbl[row][col+1] = words[0][1]
                # Check for .L.. || xL.. || .L.x
                for col in range(self.COLUMNS - 1):
                    words = []
                    if (
                        (tbl[row][col] != 0) and
                        (col == 0 or tbl[row][col-1] == 0) and
                        (tbl[row][col+1] == 0) and
                        (col+2 == self.COLUMNS or tbl[row][col+2] == 0)
                    ):
                        mCol = col+1
                        if (
                            (row == 0 or tbl[row-1][mCol] == 0) and
                            (row+1 == self.ROWS or tbl[row+1][mCol] == 0)
                        ):
                            start = tbl[row][col]
                            for word in self.word2:
                                if word[0] == start:
                                    words.append(word)
                                elif word[0] > start:
                                    break
                            if words:
                                shuffle(words)
                                tbl[row][col+1] = words[0][1]
                    # Check for ..L. || ..Lx
                    words = []
                    if (
                        (col >= 1) and
                        (tbl[row][col] != 0) and
                        (tbl[row][col-1] == 0) and
                        (col == 1 or tbl[row][col-2] == 0) and
                        (col+1 == self.COLUMNS or tbl[row][col+1] == 0) and
                        (row == 0 or tbl[row-1][col-1] == 0) and
                        (row+1 == self.ROWS or tbl[row+1][col-1] == 0)
                    ):
                        for word in self.word2:
                            if word[1] == tbl[row][col]:
                                words.append(word)
                        if words:
                            shuffle(words)
                            tbl[row][col-1] = words[0][0]

            self._flipTblMutate(tbl)
        if not self._verifyWords(tbl):
            raise Exception("Something went wrong building the puzzle")

    def _flipTbl(self, tbl):
        """Return the table rotated 90 counter clockwise."""
        return [list(line) for line in zip(*tbl[:])]

    def _flipTblMutate(self, tbl):
        """Mutate table to be rotated 90 counter clockwise."""
        rotated = self._flipTbl(tbl)
        for row in range(self.ROWS):
            for col in range(self.COLUMNS):
                tbl[row][col] = rotated[row][col]

    def _cleanPuzzle(self, puzzle):
        for row in range(self.ROWS):
            for col in range(self.COLUMNS):
                if puzzle[row][col] == 0:
                    puzzle[row][col] = '.'

    def _verifyWords(self, puzzle):
        for p in (puzzle, self._flipTbl(puzzle)):
            for line in p:
                for w in filter(None,
                                ''.join([c if c else '.' for c in line]
                                        ).split('.')):
                    if w not in self.allWordsSet:
                        return False
        return True

    def _addWordToPuzzle(self, word, puzzle):
        # Try to find a spot that has overlaps, but not a overlap
        # I.e. a letter followed by a 0
        locations = []
        lenword = len(word)
        for isRot, p in enumerate((puzzle, self._flipTbl(puzzle))):
            for row, line in enumerate(p):
                for col, c in enumerate(line[: self.COLUMNS - lenword + 1]):
                    over = 0
                    if c == 0 or c == word[0]:
                        if col == 0 or line[col - 1] == 0:
                            goodspot = True
                            for i in range(lenword):
                                if word[i] == line[col + i]:
                                    over += 1
                                elif line[col + i] != 0:
                                    goodspot = False
                                    break
                            if goodspot and over != lenword:
                                locations.append([over, isRot, row, col])
        locations.sort(key=lambda elem: elem[0], reverse=True)
        for loc in locations:
            over, isRot, row, col = loc
            # p = copy.deepcopy(puzzle)
            p = pickle.loads(pickle.dumps(puzzle, -1))
            if isRot:
                p = self._flipTbl(p)
            for i in range(lenword):
                p[row][col + i] = word[i]
            if self._verifyWords(p):
                if isRot:
                    self._flipTblMutate(puzzle)
                for i in range(lenword):
                    puzzle[row][col + i] = word[i]
                return True
        return False

    def _getWord(self, usedSet, wordsList, unusedLetters=None):
        while True:
            word = choice(wordsList)
            if unusedLetters is not None and unusedLetters.isdisjoint(word):
                continue
            elif word in usedSet:
                continue
            else:
                return word


class formatPuzzles:
    """
    Create the hints and encoding, then prepare data structures needed for PDF.

    Parameters
    ----------
    pdfSettings : dict
        The settings used to create the PDF.
    inputData : list
        The raw puzzles to be encoded.
    shortWordsPage : tuple


    Returns
    -------
    Object ready to be built.

    """

    def __init__(self, pdfSettings, inputData, shortWordsPage):
        """Initialize variables."""
        self.puzzles = []
        self.inputData = inputData
        self.pdfSettings = pdfSettings
        self.shortWordsPage = shortWordsPage

    def make(self):
        """Make the hints and encodings, then creates and opens final PDF."""
        self._readPuzzles()
        allPuzzles, allAnswers, allLeft, allHints = self._prepare()
        self._outputPuzzles(allPuzzles, allAnswers, allLeft, allHints)

    def _readPuzzles(self):
        """
        Do basic verification of puzzles.

        Optional but useful when testing new ways to make puzzles.
        """
        dataSource = self.inputData

        singlePuzzle = []
        counter = 0
        for line in dataSource:
            cleaned = line.upper().strip()
            for c in cleaned:
                if not c.isascii or not (c in ascii_uppercase
                                         or c == "."):
                    raise ValueError("Unknown char: " + c)
            if len(cleaned) == 13:
                singlePuzzle.append(cleaned)
                counter += 1
            elif len(cleaned) == 0:
                pass
            else:
                raise ValueError("Not 13 ascii chars long: " + cleaned)
            if counter == 13:
                self.puzzles.append(singlePuzzle)
                counter = 0
                singlePuzzle = []

    def _addHints(self, puzzle, decoder, encoder):
        # Ensures that there is 1 vowel in the hints, and that all hints
        # are used in the puzzle at least once.
        MIN_LIMIT = self.pdfSettings["DIFFICULTY"]
        MIN_HINTS = self.pdfSettings["MIN_HINTS"]
        hints = []  # Final output formatted for other functions
        selectedHints = []
        allLetters = [letter for line in puzzle for letter in line
                      if letter in ascii_uppercase]
        usedLetters = set(allLetters)
        unusedLetters = set(ascii_uppercase)
        unusedLetters.difference_update(usedLetters)
        vowels = {"A", "E", "I", "O", "U"}
        vowels.intersection_update(usedLetters)
        usedLetters.difference_update(vowels)
        vowels = list(vowels)
        shuffle(vowels)
        usedLetters = list(usedLetters)
        shuffle(usedLetters)
        selectedHints.append(int(encoder[vowels[0]]))

        total = len(allLetters)
        used = allLetters.count(vowels[0])

        while True:
            singleHint = usedLetters.pop()
            selectedHints.append(int(encoder[singleHint]))
            used += allLetters.count(singleHint)
            if len(selectedHints) >= MIN_HINTS and used/total > MIN_LIMIT:
                break
        # Add unused letters to hints to avoid impossible puzzles
        for letter in unusedLetters:
            selectedHints.append(int(encoder[letter]))
        for x in range(13):
            hints.append([])
            for y in range(2):
                iValue = y + 2*x + 1
                value = str(iValue).zfill(2)
                if iValue in selectedHints:
                    hints[x].append(decoder[value] + value)
                else:
                    hints[x].append(value)
        return selectedHints, hints

    def _prepare(self):
        allPuzzles = []
        allAnswers = []
        allLeft = []
        allHints = []

        for single in self.puzzles:
            letters = list(ascii_uppercase)
            shuffle(letters)
            while any([x == ord(y)-ord("A") for
                       (x, y) in zip(range(26), letters)]):
                shuffle(letters)
            encoder = {l: str(i+1).zfill(2)
                       for (i, l) in enumerate(letters)}

            decoder = {v: k for (k, v) in encoder.items()}

            allAnswers.append([[y if y.isalpha() else "00" for y in x]
                               for x in single])

            table1Left = []
            for x in range(13):
                table1Left.append([])
                for y in range(2):
                    value = str(y+2*x + 1).zfill(2)
                    table1Left[x].append(decoder[value] + value)
            allLeft.append(table1Left)

            selectedHints, hints = self._addHints(single, decoder, encoder)
            allHints.append(hints)

            puzzle = []
            for line in single:
                encLine = []
                for c in line:
                    encodedC = encoder.get(c, "0")
                    if int(encodedC) in selectedHints:
                        encLine.append(decoder[encodedC])
                    else:
                        encLine.append(encodedC.zfill(2))
                puzzle += [encLine]
            allPuzzles.append(puzzle)

        if len(self.puzzles) % 2 == 1 and not self.pdfSettings["LARGE"]:
            for item in (allPuzzles, allAnswers, allLeft, allHints):
                item.append(None)

        return allPuzzles, allAnswers, allLeft, allHints

    def _outputPuzzles(self, allPuzzles, allAnswers, allLeft, allHints):
        assert len(allPuzzles) == len(allAnswers)
        assert len(allPuzzles) == len(allLeft)
        assert len(allPuzzles) == len(allHints)
        outFile = makePDF(self.pdfSettings)
        if self.pdfSettings["LARGE"]:
            for i in range(0, len(allPuzzles)):
                outFile._addPuzzlePage(allHints[i], allPuzzles[i], True)
                outFile._addPuzzlePage(allLeft[i], allAnswers[i], False)
        else:
            for i in range(0, len(allPuzzles), 2):
                outFile._addPuzzlePage(allHints[i], allPuzzles[i], True,
                                       allHints[i+1], allPuzzles[i+1], True)
                outFile._addPuzzlePage(allLeft[i], allAnswers[i], False,
                                       allLeft[i+1], allAnswers[i+1], False)
        if self.pdfSettings["SHORT_WORDS_PAGE"]:
            outFile._addHints(self.shortWordsPage)
        outFile._finalOutput()


class makePDF:
    """
    Create and open the PDF using the systems default file handler.

    Parameters
    ----------
    pdfSettings : dict
        The settings used to create the PDF.

    """

    def __init__(self, pdfSettings):
        """Initialize variables and styles."""
        self.puzzleCount = 1  # For labels of puzzles
        self.answerCount = 1  # For labels of answers
        self._WIDTH = pdfSettings["WIDTH"]
        self._PALECOLOR = pdfSettings["PALECOLOR"]
        self._MARGIN_TOP = pdfSettings["MARGIN_TOP"]
        self._MARGIN_BOTTOM = pdfSettings["MARGIN_BOTTOM"]
        self._SHORT_WORDS_PAGE = pdfSettings["SHORT_WORDS_PAGE"]
        self._OPEN_PDF = pdfSettings["OPEN_PDF"]
        self.LARGE = pdfSettings["LARGE"]

        self.styleSheet = styles.getSampleStyleSheet()
        if self.LARGE:
            self.styleSheet.add(
                styles.ParagraphStyle(name='CenterText',
                                      parent=self.styleSheet['Normal'],
                                      alignment=enums.TA_CENTER,
                                      fontName=styles._baseFontNameI,
                                      fontSize=16)
            )
            self._STYLE_PALE = TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TEXTCOLOR', (0, 0), (-1, -1), self._PALECOLOR),
                ('FONTSIZE', (0, 0), (-1, -1), 16),
                ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
            ])
            self._STYLE_DARK = TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('FONTSIZE', (0, 0), (-1, -1), 16),
                ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
            ])
        else:
            self.styleSheet.add(
                styles.ParagraphStyle(name='CenterText',
                                      parent=self.styleSheet['Normal'],
                                      alignment=enums.TA_CENTER,
                                      fontName=styles._baseFontNameI)
            )

            self._STYLE_PALE = TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TEXTCOLOR', (0, 0), (-1, -1), self._PALECOLOR),
                ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
            ])

            self._STYLE_DARK = TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
            ])

        self._STYLE_BLANK = TableStyle([
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.white),
            ('BOX', (0, 0), (-1, -1), 0.25, colors.white),
        ])

        self.output = os.path.join(
            pdfSettings["OUTPUT_PATH"],  pdfSettings["OUTPUT_FILE"])
        self.paperSize = pdfSettings["PAPER_SIZE"]
        self.elements = []

    def _addHints(self, wordsList):
        sty = styles.ParagraphStyle(
            name='shortWordsPage',
            fontName='Courier',
            fontSize=14,
        )
        if self._SHORT_WORDS_PAGE:
            for l in wordsList:
                p = Paragraph(', '.join(l) + "<br/><br/>", style=sty)
                self.elements.append(p)

    def _finalOutput(self):
        doc = SimpleDocTemplate(self.output, pagesize=self.paperSize,
                                topMargin=self._MARGIN_TOP,
                                bottomMargin=self._MARGIN_BOTTOM)
        doc.build(self.elements)
        if self._OPEN_PDF:
            try:
                if os.name == "nt":
                    os.startfile(self.output)
                else:
                    subprocess.call(["open", self.output])
            except Exception as e:
                print("Unable to open PDF automatically.\n" + str(e))

    def _addPuzzlePage(self, table1Left, table1Center, table1Right=False,
                       table2Left=None, table2Center=None, table2Right=False):
        data = [[], []]
        t1Right = self._rightTable(table1Left) if table1Right else False
        t1Left = copy.deepcopy(table1Left)
        self._formatLeft(t1Left)

        data[0] = self._addPuzzle(t1Left, table1Center, t1Right)
        if self.LARGE:
            t1_w = 1.3 * inch
            t2_w = 14 * self._WIDTH
            t3_w = 1.5 * inch
        else:
            t2Right = self._rightTable(table2Left) if table2Right else False
            t2Left = copy.deepcopy(table2Left) if table2Left else None
            self._formatLeft(t2Left)
            data[1] = self._addPuzzle(t2Left, table2Center, t2Right)
            t1_w = 1 * inch
            t2_w = 14 * self._WIDTH
            t3_w = 1 * inch

        for i in range(1 + int(self.LARGE == 0)):
            if table1Right:
                num = f"Puzzle {str(self.puzzleCount).zfill(2)}"
                self.puzzleCount += 1
            else:
                num = f"Answer {str(self.answerCount).zfill(2)}"
                self.answerCount += 1
            if len(data[i]) == 3:
                data[i].insert(0, verticalText(num))
            else:
                data[i].insert(0, "")

        shell_table = Table(data, colWidths=[0.5 * inch, t1_w, t2_w, t3_w])
        self.elements.append(shell_table)

    def _formatLeft(self, t):
        if t is not None:
            for r, line in enumerate(t):
                for c, item in enumerate(line):
                    if len(item) == 3:
                        if self.LARGE:
                            t[r][c] = Paragraph(("<font color=grey size=8>"
                                                 f"{item[1:]}<br />"
                                                 "</font>"
                                                 "<font color=black size=12>"
                                                 f" {item[0]}</font>"),
                                                self.styleSheet["BodyText"])
                        else:
                            t[r][c] = Paragraph((f"<font color=grey>{item[1:]}"
                                                 "</font><font color=black>"
                                                 f" {item[0]}</font>"),
                                                self.styleSheet["BodyText"])

    def _rightTable(self, leftTable):
        t = []
        hints = []
        if leftTable:
            hints = [x[0] for y in leftTable for x in y if len(x) == 3]
        for x in range(13):
            t.append([])
            for y in range(2):
                letter = ascii_uppercase[y+2*x]
                if letter in hints:
                    t[x].append(Paragraph(f"<strike>{letter}</strike>",
                                          self.styleSheet["CenterText"]))
                else:
                    t[x].append(letter)
        return t

    def _addPuzzle(self, left, center, right):
        if left is None:
            t1 = Table([[""]]*13, [self._WIDTH], 13*[self._WIDTH])
            t1.setStyle(self._STYLE_BLANK)
            return [t1]

        t1 = Table(left, 2*[self._WIDTH], 13*[self._WIDTH])
        t1.setStyle(self._STYLE_PALE)

        if right:
            t3 = Table(right, 2*[self._WIDTH], 13*[self._WIDTH])
            t3.setStyle(self._STYLE_DARK)
        else:
            t3 = None

        t2 = Table(center, 13*[self._WIDTH], 13*[self._WIDTH])
        if t3:
            t2.setStyle(self._STYLE_PALE)
        else:
            t2.setStyle(self._STYLE_DARK)

        for r, line in enumerate(center):
            for c, item in enumerate(line):
                if item == "00":
                    t2.setStyle(TableStyle(
                        [('BACKGROUND', (c, r), (c, r), colors.darkgray),
                         ('TEXTCOLOR', (c, r), (c, r), colors.darkgray),
                         ]))
                elif item.isalpha():
                    t2.setStyle(TableStyle(
                        [('ALIGN', (c, r), (c, r), 'CENTER'),
                         ('VALIGN', (c, r), (c, r), 'MIDDLE'),
                         ('TEXTCOLOR', (c, r), (c, r), colors.black),
                         ]))
        return [t1, t2, t3]


class verticalText(Flowable):
    """
    Rotates and draws text.

    Parameters
    ----------
    text : str
        The text to be rotated.

    """

    def __init__(self, text):
        """Set text to print and prepares flowable to hold it."""
        Flowable.__init__(self)
        self.text = text

    def draw(self):
        """Automatic overide to rotate text."""
        canvas = self.canv
        canvas.rotate(270)
        fs = canvas._fontsize
        canvas.translate(-fs * len(self.text)/2.0, fs)
        canvas.drawString(0, 0, self.text)

    def wrap(self, aW, aH):
        """Overide to hold wrapper."""
        canv = self.canv
        fn, fs = canv._fontname, canv._fontsize
        return canv._leading, 1 + canv.stringWidth(self.text, fn, fs)


def settings(letterPaper=True, largePrint=False):
    """
    All settings needed for creating puzzles and PDF.

    Parameters
    ----------
    letterPaper : bool
        Whether to use letter size rather than A4
    largePrint : bool
        Whether to use large print (1 puzzle per page) or regular (2 per page)

    Returns
    -------
    Dict with settings.

    """
    pdfSettings = dict()
    if letterPaper:
        pdfSettings["PAPER_SIZE"] = letter
    else:
        pdfSettings["PAPER_SIZE"] = A4

    if letterPaper:
        pdfSettings["MARGIN_TOP"] = inch * .35
        pdfSettings["MARGIN_BOTTOM"] = inch * .35
    else:
        pdfSettings["MARGIN_TOP"] = inch * 0.81
        pdfSettings["MARGIN_BOTTOM"] = inch * 0.81

    pdfSettings["LARGE"] = largePrint
    if largePrint:
        pdfSettings["PAPER_SIZE"] = (
            pdfSettings["PAPER_SIZE"][1], pdfSettings["PAPER_SIZE"][0])
        width = pdfSettings["PAPER_SIZE"][0] / inch
        width = width - 1.3 - 1.5 - 0.5 - 0.5
        pdfSettings["WIDTH"] = (width/16.0) * inch
        if letterPaper:
            pdfSettings["MARGIN_TOP"] = inch * 0.8
            pdfSettings["MARGIN_BOTTOM"] = inch * 1.29
        else:
            pdfSettings["MARGIN_TOP"] = inch * 0.5
            pdfSettings["MARGIN_BOTTOM"] = inch * 0.8
    else:
        # page width - Left table - right table - label - side margins
        width = pdfSettings["PAPER_SIZE"][0] / inch
        width = width - 1.0 - 1.0 - 0.5 - 0.7
        pdfSettings["WIDTH"] = (width/14.0) * inch

    pdfSettings["PALECOLOR"] = colors.HexColor("#a1c0f0")
    pdfSettings["SHORT_WORDS_PAGE"] = False
    pdfSettings["OPEN_PDF"] = True   # Open the pdf with default file handler

    pdfSettings["DIFFICULTY"] = 0.2  # Amount of the puzzle prefilled
    pdfSettings["MIN_HINTS"] = 5     # Minimum number of letters used for hints
    pdfSettings["NUMBER_OF_PUZZLES"] = 2

    # On Windows, create PDF in My Documents
    # else, create it in the home directory.
    if os.name == "nt":
        # https://docs.microsoft.com/en-us/windows/win32/api/
        # shlobj_core/nf-shlobj_core-shgetfolderpathw
        CSIDL = 5                # My Documents
        SHGFP_TYPE_CURRENT = 0   # Get current, not default value
        buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
        ctypes.windll.shell32.SHGetFolderPathW(
            None, CSIDL, None, SHGFP_TYPE_CURRENT, buf)
        pdfSettings["OUTPUT_PATH"] = buf.value
    else:
        pdfSettings["OUTPUT_PATH"] = Path.home()
    pdfSettings["OUTPUT_FILE"] = "CluelessCrosswords.pdf"

    return pdfSettings


class GUI(tk.Frame):
    """Simple TK GUI."""

    def __init__(self, parent):
        """Expose user choices to the settings setup."""
        super(GUI, self).__init__(parent)

        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=14)

        titleFont = font.Font(family='Helvetica', size=16)
        title = tk.Label(root, text="Clueless Crosswords")
        title.configure(font=titleFont)
        title.grid(row=0, column=0, columnspan=2)

        currentRow = 1
        widgets = []

        numberLbl = tk.Label(root, text="Number of puzzles")
        self.numberSet = tk.IntVar(root)
        self.numberSet.set(2)
        numberEnt = tk.Spinbox(root, from_=1, to=20,
                               textvariable=self.numberSet,
                               state="readonly", width=4)
        numberEnt.configure(font=default_font)
        widgets.append((numberLbl, numberEnt))

        paperLbl = tk.Label(root, text="Paper size")
        self.paperSet = tk.StringVar(root)
        self.paperSet.set("letter")
        paperEnt = tk.Spinbox(root, values=("letter", "A4"),
                              textvariable=self.paperSet,
                              state="readonly", width=8)
        paperEnt.configure(font=default_font)
        widgets.append((paperLbl, paperEnt))

        difficultyLbl = tk.Label(root, text="Minimum prefilled %")
        self.difficultySet = tk.IntVar(root)
        self.difficultySet.set(20)
        difficultyEnt = tk.Spinbox(root, from_=15, to=50,
                                   textvariable=self.difficultySet,
                                   state="readonly", width=4)
        difficultyEnt.configure(font=default_font)
        widgets.append((difficultyLbl, difficultyEnt))

        largeLbl = tk.Label(root, text="Use large print")
        self.largeSet = tk.IntVar()
        largeEnt = tk.Checkbutton(root, variable=self.largeSet)
        largeEnt.select()
        widgets.append((largeLbl, largeEnt))

        openLbl = tk.Label(root, text="Auto open PDF")
        self.openSet = tk.IntVar()
        openEnt = tk.Checkbutton(root, variable=self.openSet)
        openEnt.select()
        widgets.append((openLbl, openEnt))

        shortLbl = tk.Label(root, text="Add short words page")
        self.shortSet = tk.IntVar()
        shortEnt = tk.Checkbutton(root, variable=self.shortSet)
        shortEnt.deselect()
        widgets.append((shortLbl, shortEnt))

        for lbl, wdg in widgets:
            lbl.grid(sticky=tk.W, row=currentRow, column=0, pady=4, padx=2)
            wdg.grid(sticky=tk.E, row=currentRow, column=1, pady=4, padx=2)
            currentRow += 1

        makeButton = tk.Button(root, text="Make Puzzles", command=self._make)
        makeButton.grid(row=currentRow, columnspan=2)
        currentRow += 1

        self.resultsTxt = tk.StringVar()
        self.resultsTxt.set("")
        resultsLbl = tk.Label(root, textvariable=self.resultsTxt)
        resultsLbl.grid(row=currentRow, columnspan=2, pady=4, padx=2)

    def _make(self):
        self.resultsTxt.set("Please wait.")
        root.update()
        letterPaper = self.paperSet.get() == "letter"
        largePrint = self.largeSet.get() == 1
        pdfSettings = settings(letterPaper, largePrint)
        pdfSettings["NUMBER_OF_PUZZLES"] = self.numberSet.get()
        pdfSettings["DIFFICULTY"] = self.difficultySet.get()/100.0
        pdfSettings["OPEN_PDF"] = self.openSet.get() == 1
        pdfSettings["SHORT_WORDS_PAGE"] = self.shortSet.get() == 1
        data = []
        puzzles = makePuzzles(
            pdfSettings["NUMBER_OF_PUZZLES"], data)
        shortWordsPage = puzzles.make()
        puzzleOutput = formatPuzzles(pdfSettings, data, shortWordsPage)
        puzzleOutput.make()
        self.resultsTxt.set("")
        root.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number", help="number of puzzles to make",
                        type=int, default=0)
    parser.add_argument("-w", "--words",
                        help="add a page with 2 & 3 letter words",
                        action="store_true")
    parser.add_argument("-o", "--autoopen",
                        help="open output pdf with default pdf handler",
                        action="store_true")
    parser.add_argument("-a4", "-A4", help="switch to A4 sized paper",
                        action="store_true")
    parser.add_argument("-l", "--largeprint", help="switch to large print",
                        action="store_true")

    if len(sys.argv) > 1:
        args = parser.parse_args()
        letterPaper = not args.a4
        pdfSettings = settings(letterPaper, args.largeprint)
        if args.number > 0:
            pdfSettings["NUMBER_OF_PUZZLES"] = args.number
        if args.words:
            pdfSettings["SHORT_WORDS_PAGE"] = True
        else:
            pdfSettings["SHORT_WORDS_PAGE"] = False
        if args.autoopen:
            pdfSettings["OPEN_PDF"] = True
        else:
            pdfSettings["OPEN_PDF"] = False
        data = []
        puzzles = makePuzzles(pdfSettings["NUMBER_OF_PUZZLES"], data)
        shortWordsPage = puzzles.make()
        puzzleOutput = formatPuzzles(pdfSettings, data, shortWordsPage)
        puzzleOutput.make()
    else:
        root = tk.Tk()
        root.resizable(False, False)
        main = GUI(root)
        root.title("Clueless Crosswords")
        root.mainloop()
