from flask import Blueprint, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField,SelectField,FileField,DecimalField,BooleanField
from wtforms.validators import DataRequired,Regexp, NumberRange


# Create Form Class

class inputForm(FlaskForm):
    name = StringField('User Name',validators=[DataRequired()],render_kw={'style': 'width: 30ch'})
    dataset = FileField('Load your Dataset',render_kw={'style': 'width: 30ch'},validators=[Regexp('')])
    my_intents = [('1', 'Classification'), ('2', 'Regression'), ('3', 'Clustering')]
    intent = SelectField('What is your intent?',choices=my_intents,render_kw={'style': 'width: 30ch'})
    my_metrics = [('1', 'Accuracy'), ('2', 'Precision'), ('3', 'Recall')]
    multiple = SelectField('Metric to optimize?',choices = my_metrics, default = ['1'],render_kw={'style': 'width: 30ch'})
    time = DecimalField('Time limit (in seconds)',validators=[DataRequired(),NumberRange(0,10)],render_kw={'style': 'width: 30ch'})
    prepro = BooleanField('Is preprocessing needed?',render_kw={'style': 'width: 30ch'})
    my_algorithms = [('1', 'SVM'), ('2', 'KNN'), ('3', 'RF')]
    algorithm = SelectField('Would you like a particular algorithm?',choices = my_algorithms, default = ['RF'],render_kw={'style': 'width: 30ch'})

    submit = SubmitField('Submit')


views = Blueprint('views', __name__)


@views.route('/', methods=['GET', 'POST'])
def home():
    form = inputForm()
    if request.method == 'GET':
        return render_template("home.html",
                               form = form)
    
    elif request.method == 'POST':
        print('Here')
        name = form.name.data
        intent = form.intent.data
        dataset = form.dataset.data
        print(name,intent,dataset)

        return render_template('after.html')




