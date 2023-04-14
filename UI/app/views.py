from flask import Blueprint, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField,SelectField,FileField,DecimalField,BooleanField
from wtforms.validators import DataRequired, NumberRange


# Create Form Class

class inputForm(FlaskForm):
    name = StringField('User Name',validators=[DataRequired()],render_kw={'style': 'width: 30ch'})
    #dataset = FileField('Load your Dataset',render_kw={'style': 'width: 30ch'},validators=[Regexp('')])
    my_datasets = [('iris', 'iris'), ('wines', 'wines'), ('boston', 'boston')]
    dataset = SelectField('What is your dataset?',choices=my_datasets,render_kw={'style': 'width: 30ch'})
    my_intents = [('classification', 'Classification'), ('regression', 'Regression'), ('clustering', 'Clustering')]
    intent = SelectField('What is your intent?',choices=my_intents,render_kw={'style': 'width: 30ch'})
    my_metrics = [('acc', 'Accuracy'), ('pre', 'Precision'), ('rec', 'Recall')]
    metric = SelectField('Metric to optimize?',choices = my_metrics,render_kw={'style': 'width: 30ch'})
    time = DecimalField('Time limit (in seconds)',validators=[DataRequired(),NumberRange(0,3600)],render_kw={'style': 'width: 30ch'})
    prepro = BooleanField('Is Preprocessing needed?',render_kw={'style': 'width: 30ch'})
    my_algorithms = [('svm', 'SVM'), ('knn', 'KNN'), ('rf', 'RF')]
    algorithm = SelectField('Would you like a particular algorithm?',choices = my_algorithms,render_kw={'style': 'width: 30ch'})

    submit = SubmitField('Submit')


views = Blueprint('views', __name__)


@views.route('/', methods=['GET', 'POST'])
def home():
    form = inputForm(algorithm='svm')
    if request.method == 'GET':
        return render_template("home.html",
                               form = form)
    
    elif request.method == 'POST':
        data = {'User':form.name.data,'Intent':form.intent.data,'Dataset':form.dataset.data,'Time':form.time.data,
                'Metric':form.metric.data,'Preprocessing':form.prepro.data,'algorithm':form.algorithm.data}

        print(data)

        return render_template('after.html')




