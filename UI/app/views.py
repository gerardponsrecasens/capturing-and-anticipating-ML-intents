from flask import Blueprint, render_template, request, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField,SelectField,FileField,DecimalField,BooleanField,TextAreaField,RadioField
from wtforms.validators import DataRequired, NumberRange
from .generate_pipeline import pipeline_generator
from .generate_triples import generate_user_dataset, generate_intent, generate_all
from .recommend import recommendation
from .queries import get_algorithm, get_intent, get_metric,get_preprocessing,get_preprocessing_algorithm


# Create Form Class for the User Input. They are structured so that the recommendation engine
# can make use of the already filled inputs.

class initialForm(FlaskForm):
    '''
    First intital form where the user can select the datasets that he/she wants to use.
    '''
    my_anticipations = [('kge','Link Prediction'),('query','SPARQL')]
    anticipation = SelectField('Anticipation Method',choices=my_anticipations,render_kw={'style': 'width: 30ch'})
    name = StringField('User Name',validators=[DataRequired()],render_kw={'style': 'width: 30ch'})
    my_datasets = [('iris', 'iris'), ('wines', 'wines'), ('boston', 'boston')]
    dataset = SelectField('What is your dataset?',choices=my_datasets,render_kw={'style': 'width: 30ch'})
    submit = SubmitField('Proceed')

class intentForm(FlaskForm):
    '''
    Second form where the user specifies the Intent
    '''
    name = StringField('User Name',validators=[DataRequired()],render_kw={'style': 'width: 30ch',"readonly": True})
    my_anticipations = [('kge','Link Prediction'),('query','SPARQL')]
    anticipation = SelectField('Anticipation Method',choices=my_anticipations,render_kw={'style': 'width: 30ch',"readonly": True})
    my_datasets = [('iris', 'iris'), ('wines', 'wines'), ('boston', 'boston')]
    dataset = SelectField('What is your dataset?',choices=my_datasets,render_kw={'style': 'width: 30ch',"readonly": True})
    my_intents = [('Classification', 'Classification'), ('Regression', 'Regression'), ('Clustering', 'Clustering')]
    intent = SelectField('What is your intent?',choices=my_intents,render_kw={'style': 'width: 30ch'})
    submit = SubmitField('Proceed')

class inputForm(FlaskForm):
    '''
    Final form where the user specifies the evaluation requirements and constraints.
    '''
    name = StringField('User Name',validators=[DataRequired()],render_kw={'style': 'width: 30ch',"readonly": True})
    my_anticipations = [('kge','Link Prediction'),('query','SPARQL')]
    anticipation = SelectField('Anticipation Method',choices=my_anticipations,render_kw={'style': 'width: 30ch',"readonly": True})
    my_datasets = [('iris', 'iris'), ('wines', 'wines'), ('boston', 'boston')]
    dataset = SelectField('What is your dataset?',choices=my_datasets,render_kw={'style': 'width: 30ch',"readonly": True})
    my_intents = [('Classification', 'Classification'), ('Regression', 'Regression'), ('Clustering', 'Clustering')]
    intent = SelectField('What is your intent?',choices=my_intents,render_kw={'style': 'width: 30ch',"readonly": True})
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

class ratingForm(FlaskForm):
    '''
    After the results are presented, the users can give their feedback (rating and comment)
    '''
    rating = RadioField('How would you rate this Workflow?', choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5')], default='3')
    feedback = TextAreaField('Leave any comments you may have')
    submit = SubmitField('Submit')



views = Blueprint('views', __name__)


'''
General views. On it, the user gets asked to provide the input on the 'GET' phase. Then, when the user
submits the form ('POST' phase), the system generates recommendations or the workflow. Finally, the 
user is presented the final solution, and the feedback is stored
'''

# Ask for the user Name and the Dataset

@views.route('/', methods=['GET', 'POST'])
def home():
    form = initialForm()

    # If the user is logged, the same name is used

    if request.method == 'GET' and 'User' in session:
        form.name.data = session['User']

    # First time the user is using the system

    if request.method == 'GET':
        return render_template("initial.html",
                               form = form)
    

    elif request.method == 'POST':

        print(form.name.data)

        # Once the dataset has been specified, the system variables are stored and the first triples are generated

        session['Anticipation'] = form.anticipation.data
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
    form.anticipation.data = session['Anticipation']

    print(session['User'],session['Dataset'])
    
    if request.method == 'GET':

        if session['Anticipation'] == 'query':
            form.intent.data = get_intent(session['User'],session['Dataset']).split('#')[1]
            return render_template("intent.html",
                                form = form)
        else:
            form.intent.data = recommendation(stage=1, task = session['Task'])
            return render_template("intent.html",
                                form = form)
    
    elif request.method == 'POST':

        session['Intent'] = form.intent.data
        generate_intent(user = session['User'], dataset = session['Dataset'],
                        user_intent = session['Intent'], task = session['Task'],
                        current_time = session['current_time'])
        

        return redirect(url_for('views.eval_const'))



# Ask for the User Requirements and Constraints
@views.route('/eval_const', methods=['GET', 'POST'])
def eval_const():
    form = inputForm()
    form.name.data = session['User']
    form.dataset.data = session['Dataset']
    form.intent.data = session['Intent']
    form.anticipation.data = session['Anticipation']

    if request.method == 'GET':

        if session['Anticipation'] == 'query':
            form.time.data = int(20)
            form.algorithm.data = get_algorithm(session['User'],session['Dataset'],session['Intent']).split('-')[1]
            form.prepro.data = get_preprocessing(session['User'],session['Dataset'],session['Intent'])
            form.preprocessor.data = get_preprocessing_algorithm(session['User'],session['Dataset'],session['Intent']).split('-')[1]
            form.metric.data = get_metric(session['User'],session['Dataset'],session['Intent']).split('#')[1]

            return render_template("complete.html",
                               form = form)

        else:
            algorithm_constraint, prepro_constraint, metric = recommendation(stage = 2, task = session['Task'])
            
                        
            form.algorithm.data = algorithm_constraint

            if prepro_constraint == 'NoPre':
                form.prepro.data = False
            else:
                form.prepro.data = True
                form.preprocessor.data = prepro_constraint
            
            form.metric.data = metric
            form.time.data = int(20)

            return render_template("complete.html",
                                form = form)
    
    elif request.method == 'POST':

        data = {'User':form.name.data,'Intent':form.intent.data,'Dataset':form.dataset.data,'Time':float(form.time.data),
                'Metric':form.metric.data,'Preprocessing':form.prepro.data,'Algorithm':form.algorithm.data,
                'PreproAlgorithm':form.preprocessor.data, 
                'Hyperparameter':form.hyperparam.data if form.hyperparam.data != 'None' else None,
                'Hyperparameter_value': int(form.hyperparam_value.data) if form.hyperparam_value.data is not None else None}
        
        print(data)

        session['metric'] = form.metric.data
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
                               form = feedback,data = session['score'], metric = session['metric'])
    
    elif request.method == 'POST':

        generate_all(feedback=[feedback.rating.data,feedback.feedback.data],current_time = session['current_time'],
                      user = session['User'],dataset = session['Dataset'],
                      workflow = session['Workflow'],task = session['Task'])
        
        #TO DO: incorporate COMMENT in feedback

        return redirect(url_for('views.home'))




