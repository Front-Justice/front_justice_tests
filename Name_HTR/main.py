import glob
import subprocess
import easyocr



if __name__ == '__main__':


	reader = easyocr.Reader(['fr'])
	result = reader.readtext('jacquot.png')
	for res in result:
		print(res)
