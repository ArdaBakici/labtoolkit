from flask import Flask, redirect, render_template, url_for, request, Response, after_this_request, session
from werkzeug.utils import secure_filename
import os
import cv2
from processing import process
import base64
import io 

app = Flask(__name__)
app.secret_key = "XaBhgkdj&sJs2s!5df!849"
uploads_dir = os.path.join(app.root_path, 'uploads')
results_dir = os.path.join(app.root_path, 'results')
ALLOWED_EXTENSIONS = {'bmp', 'dib', 'jpg', 'jpeg', 'jpe', 'jp2', 'png', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'tiff', 'tif'}
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)


def get_base64_encoded_image(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
	return redirect(url_for('upload', lang='tr'))


@app.route('/<lang>/about')
def about(lang):
	if lang == 'tr':
		return render_template('about_tr.html')
	elif lang == 'en':
		return render_template('about_en.html')


@app.route('/<lang>/upload')
def upload(lang):
	if lang == 'tr':
		return render_template('index_tr.html')
	elif lang == 'en':
		return render_template('index_en.html')


@app.route('/<lang>/processing', methods=['GET', 'POST'])
def processing(lang):
	if request.method == 'GET':
		return redirect(url_for('upload', lang=lang))  #Accessed using GET method
	image = request.files.get('image', None)
	if image is None:
		return redirect(url_for('upload', lang=lang))  # There is no image
	elif allowed_file(image.filename):
		filename = secure_filename(image.filename)
		path = os.path.join(uploads_dir, filename)
		session['resultpath'] = os.path.join(results_dir, f'results_{filename}')
		blursize = request.form.get('blursize', None)
		kernelsize = request.form.get('kernelsize', None)
		minArea = request.form.get('minArea', None)
		epsilon = request.form.get('epsilon', None)
		deadcellThresh = request.form.get('deadcellThresh', None)
		deadcellKernelSize = request.form.get('deadcellKernelSize', None)
		session['dilutionFactor'] = request.form.get('dilutionFactor', None)
		if session['dilutionFactor'] == '':
			session['dilutionFactor'] = -1
		else:
			session['dilutionFactor'] = float(session['dilutionFactor'])
			if session['dilutionFactor'] < 0:
				return redirect(url_for('upload', lang=lang))  # Invalid value
		image.save(path)
		try:
			session['livecell_number'], session['deadcell_number'] = gen(path, kernel_size=kernelsize, min_area=minArea, 
																		 min_blursize=blursize,
																		 epsilon=epsilon,
																		 deadcell_thresh=deadcellThresh,
																		 deadcell_kernel_size=deadcellKernelSize)
		except Exception as e:
			print(e)
			return redirect(url_for('upload', lang=lang))
		return redirect(url_for('results', lang=lang))
	else:
		return redirect(url_for('upload', lang=lang))  # Not allowed filename


@app.route('/<lang>/results')
def results(lang):
	if session['resultpath'] is None:
		return redirect(url_for('upload', lang=lang))
	else:
		bytesio = io.BytesIO()
		bytesio = get_base64_encoded_image(session['resultpath'])
		os.remove(session['resultpath'])
		session['resultpath'] = None
		if lang == 'en':
			return render_template('results_en.html', img= bytesio, alivecell_count= session['livecell_number'], deadcell_count=session['deadcell_number'], dilutionFactor=session['dilutionFactor'])
		elif lang == 'tr':
			return render_template('results_tr.html', img= bytesio, alivecell_count= session['livecell_number'], deadcell_count=session['deadcell_number'], dilutionFactor=session['dilutionFactor'])


def gen(filepath, *, kernel_size=None, min_area=None, min_blursize=None, epsilon=None, deadcell_thresh=None, deadcell_kernel_size=None):
	img = cv2.imread(filepath)
	result, livecell_amount, deadcell_amount = process(img, kernelSize=kernel_size, _minArea=min_area,
													   _blursize=min_blursize, _epsilon=epsilon,
													   _deadcellThresh=deadcell_thresh, _deadKernelSize=deadcell_kernel_size)
	os.remove(filepath)
	session['resultpath'], extension = session['resultpath'].rsplit('.', 1)
	session['resultpath'] = f"{session['resultpath']}.png"
	cv2.imwrite(session['resultpath'], result)
	return livecell_amount, deadcell_amount


if __name__ == '__main__':
	app.run(debug=True)