from flask import Blueprint, render_template, request, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField,SelectField,FileField,DecimalField,BooleanField,TextAreaField,RadioField
from wtforms.validators import DataRequired, NumberRange
from .generate_pipeline import generate
from .generate_triples import generate_triples


# Create Form Class for the User Input

class inputForm(FlaskForm):
    name = StringField('User Name',validators=[DataRequired()],render_kw={'style': 'width: 30ch'})
    #dataset = FileField('Load your Dataset',render_kw={'style': 'width: 30ch'},validators=[Regexp('')])
    my_datasets = [('iris', 'iris'), ('wines', 'wines'), ('boston', 'boston')]
    dataset = SelectField('What is your dataset?',choices=my_datasets,render_kw={'style': 'width: 30ch'})
    my_intents = [('Classification', 'Classification'), ('Regression', 'Regression'), ('Clustering', 'Clustering')]
    intent = SelectField('What is your intent?',choices=my_intents,render_kw={'style': 'width: 30ch'})
    my_metrics = [('Accuracy', 'Accuracy'), ('Precision', 'Precision'), ('F1', 'F1'), ('AUC','AUC')]
    metric = SelectField('Metric to optimize?',choices = my_metrics,render_kw={'style': 'width: 30ch'})
    time = DecimalField('Time limit (in seconds)',validators=[DataRequired(),NumberRange(0,3600)],render_kw={'style': 'width: 30ch'})
    prepro = BooleanField('Is Preprocessing needed?',render_kw={'style': 'width: 30ch'})
    my_algorithms = [('SVC', 'SVC'), ('KNeighborsClassifier', 'KNeighborsClassifier'), 
                     ('RandomForestClassifier', 'RandomForestClassifier'),('LogisticRegression','LogisticRegression')]
    algorithm = SelectField('Would you like a particular algorithm?',choices = my_algorithms,render_kw={'style': 'width: 30ch'})

    submit = SubmitField('Submit')

# Create Form Class for the User Ratings

class ratingForm(FlaskForm):
    #my_ratings = [(1, 1), (2, 2),(3, 3),(4, 4),(5, 5)]
    #rating = SelectField('How would you rate this Workflow?',choices=my_ratings,render_kw={'style': 'width: 30ch'})
    rating = RadioField('How would you rate this Workflow?', choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5')], default='3')
    feedback = TextAreaField('Leave any comments you may have')
    submit = SubmitField('Submit')


views = Blueprint('views', __name__)


'''
General view. On it, the user gets asked to provide the input on the 'GET' phase. Then, when the user
submits the form ('POST' phase), the system automatically finds a Workflow satisfiying the constraints
and computes the score. The scores gets stored in the system, and the user is redirected to the 
Feedback screen. 
'''
@views.route('/', methods=['GET', 'POST'])
def home():
    form = inputForm(algorithm='SVC')
    if request.method == 'GET':
        return render_template("home.html",
                               form = form)
    
    elif request.method == 'POST':

        data = {'User':form.name.data,'Intent':form.intent.data,'Dataset':form.dataset.data,'Time':float(form.time.data),
                'Metric':form.metric.data,'Preprocessing':form.prepro.data,'Algorithm':form.algorithm.data}
        print(data)
        score = generate(data)
        session['score'] = score

        return redirect(url_for('views.feedback_screen'))


'''
Feedback view. The user gets shown the results of the generated pipelines, and he/she can leave a review,
in terms of rating and of a comment. Upon submission, it gets redirected to the main screen, where he/she
can launch another query Before that, the generated workflow gets annotated.
'''

@views.route('/feedback_screen', methods=['GET', 'POST'])
def feedback_screen():
    feedback = ratingForm()
    if request.method == 'GET':
        return render_template("after.html",
                               form = feedback,data = session['score'])
    
    elif request.method == 'POST':
        print(feedback.rating.data,feedback.feedback.data)
        generate_triples([feedback.rating.data,feedback.feedback.data]) #TO DO: incorporate COMMENT in feedback

        return redirect(url_for('views.home'))




