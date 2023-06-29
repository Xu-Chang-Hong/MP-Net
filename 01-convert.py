# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:48:04 2022
@author: XCH
"""

from PIL import Image
I = Image.open(r'.\03.png')       
#I.show()
L = I.convert('L')
#L.show()
L.save(r'.\033.png')
