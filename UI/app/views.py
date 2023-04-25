from flask import Blueprint, render_template, request, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField,SelectField,FileField,DecimalField,BooleanField,TextAreaField,RadioField
from wtforms.validators import DataRequired, NumberRange
from .generate_pipeline import pipeline_generator
from .generate_triples import generate_triples, generate_user_dataset, generate_intent, generate_rest
from .recommend import recommendation


# Create Form Class for the User Input. They are structured so that the recommendation engine
# can make use of the already filled inputs.

class initialForm(FlaskForm):
    name = StringField('User Name',validators=[DataRequired()],render_kw={'style': 'width: 30ch'})
    my_datasets = [('iris', 'iris'), ('wines', 'wines'), ('boston', 'boston')]
    dataset = SelectField('What is your dataset?',choices=my_datasets,render_kw={'style': 'width: 30ch'})
    submit = SubmitField('Proceed')

class intentForm(FlaskForm):
    name = StringField('User Name',validators=[DataRequired()],render_kw={'style': 'width: 30ch'})
    my_datasets = [('iris', 'iris'), ('wines', 'wines'), ('boston', 'boston')]
    dataset = SelectField('What is your dataset?',choices=my_datasets,render_kw={'style': 'width: 30ch'})
    my_intents = [('Classification', 'Classification'), ('Regression', 'Regression'), ('Clustering', 'Clustering')]
    intent = SelectField('What is your intent?',choices=my_intents,render_kw={'style': 'width: 30ch'})
    submit = SubmitField('Proceed')

class inputForm(FlaskForm):
    name = StringField('User Name',validators=[DataRequired()],render_kw={'style': 'width: 30ch'})
    my_datasets = [('iris', 'iris'), ('wines', 'wines'), ('boston', 'boston')]
    dataset = SelectField('What is your dataset?',choices=my_datasets,render_kw={'style': 'width: 30ch'})
    my_intents = [('Classification', 'Classification'), ('Regression', 'Regression'), ('Clustering', 'Clustering')]
    intent = SelectField('What is your intent?',choices=my_intents,render_kw={'style': 'width: 30ch'})
    my_metrics = [('Accuracy', 'Accuracy'), ('Precision', 'Precision'), ('F1', 'F1'), ('AUC','AUC')]
    metric = SelectField('Metric to optimize?',choices = my_metrics,render_kw={'style': 'width: 30ch'})
    time = DecimalField('Time limit (in seconds)',validators=[DataRequired(),NumberRange(0,3600)],render_kw={'style': 'width: 30ch'})
    prepro = BooleanField('Is Preprocessing needed?',render_kw={'style': 'width: 30ch'})
    my_preprocessors = [('StandardScaler', 'StandardScaler'), ('MinMaxScaler', 'MinMaxScaler'),('Normalizer', 'Normalizer'), ('Any','Any')]
    preprocessor = SelectField('Restrict Preprocessing Algorithm?',choices = my_preprocessors,render_kw={'style': 'width: 30ch'})
    my_algorithms = [('SVC', 'SVC'), ('KNeighborsClassifier', 'KNeighborsClassifier'), 
                     ('RandomForestClassifier', 'RandomForestClassifier'),('LogisticRegression','LogisticRegression'),('Any','Any')]
    algorithm = SelectField('Restrict Algorithm?',choices = my_algorithms,render_kw={'style': 'width: 30ch'})
    hyperparam = SelectField('Restrict Hyperparameter?',choices=[('', 'Select an option')],render_kw={'style': 'width: 30ch'})
    hyperparam_value = DecimalField('Hyperparameter Value',render_kw={'style': 'width: 30ch'})

    submit = SubmitField('Submit')

# Create Form Class for the User Ratings

class ratingForm(FlaskForm):
    rating = RadioField('How would you rate this Workflow?', choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5')], default='3')
    feedback = TextAreaField('Leave any comments you may have')
    submit = SubmitField('Submit')



views = Blueprint('views', __name__)


'''
General views. On it, the user gets asked to provide the input on the 'GET' phase. Then, when the user
submits the form ('POST' phase), the system automatically finds a Workflow satisfiying the constraints
and computes the score. The scores gets stored in the system, and the user is redirected to the 
Feedback screen. 
'''

# Ask for the user Name and the Dataset

@views.route('/', methods=['GET', 'POST'])
def home():
    form = initialForm()
    if request.method == 'GET' and 'User' in session:
        form.name.data = session['User']

    if request.method == 'GET':
        return render_template("initial.html",
                               form = form)
    
    elif request.method == 'POST':

        print(form.name.data)
        session['User'] = form.name.data
        session['Dataset'] = form.dataset.data

        workflow,task,current_time = generate_user_dataset(session['User'],session['Dataset'])
        session['Workflow'] = workflow
        session['Task'] = task
        session['current_time'] = current_time

        print(session['User'],session['Dataset'])


        return redirect(url_for('views.intent'))
    

# Ask for the User Intent
@views.route('/intent', methods=['GET', 'POST'])
def intent():
    form = intentForm()
    form.name.data = session['User']
    form.dataset.data = session['Dataset']
    if request.method == 'GET':
        form.intent.data = recommendation(stage=1, task = session['Task'])
        return render_template("intent.html",
                               form = form)
    
    elif request.method == 'POST':

        session['Intent'] = form.intent.data
        evalRequirement,algoConst = generate_intent(user = session['User'], dataset = session['Dataset'],
                        user_intent = session['Intent'], task = session['Task'],
                        current_time = session['current_time'])
        
        session['evalRequirement'] = evalRequirement
        session['algoConst'] = algoConst

        return redirect(url_for('views.eval_const'))



# Ask for the User Requirements and Constraints
@views.route('/eval_const', methods=['GET', 'POST'])
def eval_const():
    form = inputForm()
    form.name.data = session['User']
    form.dataset.data = session['Dataset']
    form.intent.data = session['Intent']

    if request.method == 'GET':

        algorithm_constraint, prepro_constraint, metric = recommendation(stage = 2, task = session['Task'], 
                                                                 evalRequirement = session['evalRequirement'], 
                                                                 algoConst = session['algoConst'])
        
        form.algorithm.data = algorithm_constraint

        if prepro_constraint == 'NoPre':
            form.prepro.data = False
        else:
            form.prepro.data = True
            form.preprocessor.data = prepro_constraint
        
        form.metric.data = metric
        form.time.data = int(20)

        return render_template("home.html",
                               form = form)
    
    elif request.method == 'POST':

        data = {'User':form.name.data,'Intent':form.intent.data,'Dataset':form.dataset.data,'Time':float(form.time.data),
                'Metric':form.metric.data,'Preprocessing':form.prepro.data,'Algorithm':form.algorithm.data,
                'PreproAlgorithm':form.preprocessor.data, 'Hyperparameter':form.hyperparam.data,
                'Hyperparameter_value': int(form.hyperparam_value.data)}


        score = pipeline_generator(data)
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
        return render_template("feedback.html",
                               form = feedback,data = session['score'])
    
    elif request.method == 'POST':

        generate_rest(feedback=[feedback.rating.data,feedback.feedback.data],current_time = session['current_time'],
                      user = session['User'],dataset = session['Dataset'],
                      workflow = session['Workflow'],task = session['Task'],
                      evalRequirement = session['evalRequirement'],algoConst = session['algoConst'])
        #TO DO: incorporate COMMENT in feedback

        return redirect(url_for('views.home'))




