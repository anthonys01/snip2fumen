import unittest
from snip2fumen.recog import BoardRecognizer, FumenEncoder


class RegogTests(unittest.TestCase):
    def test_jstris1(self):
        br = BoardRecognizer()
        g = br.recognize_file("./img/jstris1.png")
        self.assertEqual(FumenEncoder.to_fumen(g),
                         "http://fumen.zui.jp/?v115@teR4GeR4PfwwBeRpDexwTpBei0wwRpilBewhg0R4At?"
                         "glBtBeglxhRpglAtCehlwhRpR4Aei0Btwwg0glQ4AeBtg0Q?"
                         "4ywglCeBtR4RphlAezhQ4RpJeAgH"
                         )

    def test_jstris2(self):
        br = BoardRecognizer()
        g = br.recognize_file("./img/jstris2.png")
        self.assertEqual(FumenEncoder.to_fumen(g),
                         "http://fumen.zui.jp/?v115@1eRpHeRp5ewwHeywGeg0DeRpCei0BeRpCeAtEeglQ4?"
                         "AeBtwhAeQ4ilR4BtwhAeR4RpxwBtwhglg0Q4AeglywAtwhR?"
                         "4Beg0RpBtR4Ceg0RpAtzhAeh0JeAgH"
                         )

    def test_jstris3(self):
        br = BoardRecognizer()
        g = br.recognize_file("./img/jstris3.png")
        self.assertEqual(FumenEncoder.to_fumen(g),
                         "http://fumen.zui.jp/?v115@4ewwIexwHewwfehlHewhglAeR4EewhglR4BtDewhRp?"
                         "DtBeg0whRpglBtCeg0wwjlRpAeh0xwRpglRpAeg0whwwQ4R?"
                         "phlwwAeg0whH8AeI8AeI8AeI8AeA8JeAgH"
                         )

    def test_fourtris1(self):
        br = BoardRecognizer()
        g = br.recognize_file("./img/fourtris1.png")
        self.assertEqual(FumenEncoder.to_fumen(g),
                         "http://fumen.zui.jp/?v115@zeAtGeglBtGeglAtFeQ4AeklCeR4BtglRpBehlR4Bt?"
                         "RpCeglg0R4g0zhAeglg0whQ4i0AtBeh0whywBtBeRpwhQ4x?"
                         "wAtCeRphlQ4wwR4Aeg0BtwhglRpBtAeg0AtxhglRpAeBtR4?"
                         "xhg0wwBeS4glxhg0xwAtR4AeglwhglRpAtxwAei0F8AeI8A?eC8JeAgH"
                         )

    def test_fourtris2(self):
        br = BoardRecognizer()
        g = br.recognize_file("./img/fourtris2.png")
        self.assertEqual(FumenEncoder.to_fumen(g),
                         "http://fumen.zui.jp/?v115@cfg0Iei0DeRpCeglDeRpAeilEeAtCeh0AewwAeBtCe?"
                         "g0AeywAtDeg0BeR4whDeQ4AeR4AewhDeR4Aeh0whEeQ4Aeg?"
                         "0AewhwwDeglAeg0AeywAeilBtAeRpBewwCeBtRpAeywzhJe?AgH"
                         )

    def test_tetrio1(self):
        br = BoardRecognizer()
        g = br.recognize_file("./img/tetrio1.png")
        self.assertEqual(FumenEncoder.to_fumen(g),
                         "http://fumen.zui.jp/?v115@SfilRpEeglBtRpwwBeR4xhBtywRpAexhi0BtRpAegl?"
                         "zhAtwwBtAeglAei0xwAeBthlRpg0Q4wwBeglh0RpAtR4Beg?"
                         "lg0RpAtywilAezhR4Beg0DeR4Cei0BeBtEeR4BeBtRpAeR4?CewwAeRpAezhywJeAgH"
                         )

    def test_tetrio2(self):
        br = BoardRecognizer()
        g = br.recognize_file("./img/tetrio2.png")
        self.assertEqual(FumenEncoder.to_fumen(g),
                         "http://fumen.zui.jp/?v115@ZfwhIewhCeRpDewhCeRpBtBewhCeywBtAeBtCewwil?"
                         "AewhBtAeili0whQ4Ceglh0Q4g0whR4Aewwglg0T4i0wwwhR?"
                         "pQ4AeglywBtRpwwAehlwwzhxwAeG8AeH8AeC8AeI8JeAgH"
                         )

    def test_small_sc1(self):
        br = BoardRecognizer()
        g = br.recognize_file("./img/smallsc1.png")
        self.assertEqual(FumenEncoder.to_fumen(g),
                         "http://fumen.zui.jp/?v115@FhAtg0GeBtg0EeR4Ath0DeR4zhJeAgH"
                         )


if __name__ == '__main__':
    unittest.main()
