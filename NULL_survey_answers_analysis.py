import lxml.etree as etree
import ast
from itertools import chain, combinations


#Load the survey answer dataset and parse it as a etree.
def parse_survey_answer(path):
    return etree.parse(path)


#BASIC FILTER FUNCTION!

#Input the answer tree and return a the answer tree of such that the answer to questionID is equal to output
def question_output_filter_equal(answerTree, questionID, output):
    filteredResponses = etree.fromstring(etree.tostring(answerTree))
    for response in filteredResponses.findall('Response'):
        questionElement = response.find(questionID)
        if questionElement.text != output:
            filteredResponses.remove(response)
    return filteredResponses


#Input two answer trees and return the union of them
def filter_union(answerTreeA, answerTreeB):
    filteredResponses = etree.fromstring(etree.tostring(answerTreeA))
    record_ids = set()
    for response in answerTreeA.findall('Response'):
        record_ids.add(response.find('_recordId').text)
    for response in answerTreeB.findall('Response'):
        if response.find('_recordId').text not in record_ids:
            filteredResponses.append(response)
    return filteredResponses

#Input two answer trees A and B and return the answers in A which are not in B.
def filter_minus(answerTreeA, answerTreeB):
    filteredResponses = etree.fromstring(etree.tostring(answerTreeA))
    record_ids = set()
    for response in answerTreeB.findall('Response'):
        record_ids.add(response.find('_recordId').text)
    for response in filteredResponses.findall('Response'):
        if response.find('_recordId').text in record_ids:
            filteredResponses.remove(response)
    return filteredResponses




#Demographic Filters.

#Input the answer tree and return a the answer tree of academic only.
def academic_filter(answerTree):
    studentResponses = question_output_filter_equal(answerTree,'QID224','Student')
    profesorResponses = question_output_filter_equal(answerTree,'QID224','Professor')
    educationResponses = question_output_filter_equal(answerTree,'QID219','Education')
    filteredResponses = filter_union(studentResponses,profesorResponses)
    filteredResponses = filter_union(filteredResponses,educationResponses)
    return filteredResponses

#Input the answer tree and return a the answer tree of practitioners only.
def practitioners_filter(answerTree):
    studentResponses = question_output_filter_equal(answerTree,'QID224','Student')
    profesorResponses = question_output_filter_equal(answerTree,'QID224','Professor')
    educationResponses = question_output_filter_equal(answerTree,'QID219','Education')
    filteredResponses = filter_union(studentResponses,profesorResponses)
    filteredResponses = filter_union(filteredResponses,educationResponses)
    return filter_minus(answerTree, filteredResponses)

#Input the answer tree and return a the answer tree of backend users only.
def backend_filter(answerTree):
    return filter_union(question_output_filter_equal(answerTree,'QID171','Often'),question_output_filter_equal(answerTree,'QID171','Regularly'))

#Input the answer tree and return a the answer tree of frontend users only.
def backend_filter(answerTree):
    return filter_union(filter_union(question_output_filter_equal(answerTree,'QID171','Never'),question_output_filter_equal(answerTree,'QID171','Infrequently')),question_output_filter_equal(answerTree,'QID171','Occasionally'))


#Involvement data analysis

#Id of the timing question
timing_question_ids = ['QID240', 'QID123', 'QID172', 'QID124', 'QID125', 'QID126', 'QID127', 'QID128', 'QID129', 'QID130', 'QID131', 'QID132', 'QID133', 'QID180', 'QID208', 'QID210', 'QID174', 'QID182', 'QID142', 'QID143', 'QID188', 'QID189', 'QID146', 'QID147', 'QID237', 'QID148', 'QID154', 'QID155', 'QID158', 'QID159', 'QID202', 'QID199', 'QID161', 'QID162', 'QID163', 'QID230']

#Remove the respondent from which the time spend is 0 : Question not displayed
def time_question_filter(answerTree, pageID):
    filteredResponses = etree.fromstring(etree.tostring(answerTree))
    for response in filteredResponses.findall('Response'):
        questionElement = response.find(pageID+'_PAGE_SUBMIT')
        if questionElement == None:
            filteredResponses.remove(response)
            continue
        if questionElement.text == None or questionElement.text == '' or float(questionElement.text) <= 0 :
            filteredResponses.remove(response)
    return filteredResponses


#Input an answer tree and a timing question id and return the average timer spent on the question.
def average_time_spend(answerTree,pageID):
    i = 0.0
    count = 0.0
    for response in answerTree.findall('Response'):
        questionElement = response.find(pageID+'_PAGE_SUBMIT')
        if questionElement.text == None:
            continue
        if float(questionElement.text) > 3600:
            continue
        i+=1.0
        time = float(questionElement.text)
        count += time

    return count / i


#Input an answer tree and return the cumulative time spend and the number of participant for each question.
def cumulative_average_time_spend(answerTree):
    list_average_time_spend = [0]
    number_of_respondent = []
    filteredResponses = etree.fromstring(etree.tostring(answerTree))
    for pageID in timing_question_ids:
        list_average_time_spend += [list_average_time_spend[-1]+average_time_spend(filteredResponses,pageID)]
        filteredResponses = time_question_filter(filteredResponses,pageID)
        number_of_respondent += [len(list(filteredResponses))]
    return (list_average_time_spend,number_of_respondent)


#Statistics output


#Input an answer tree and single-choice or frequency-likert scale question ID and return statistics about it.
def compute_singletextaggscore(answerTree,questionID):
    agg_count = {}
    for response in answerTree.findall('Response'):
        questionElement = response.find(questionID)
        if questionElement == None:
            continue
        elif questionElement.text == None:
            continue
        elif questionElement.text in agg_count.keys():
            agg_count[questionElement.text] += 1
        else:
            agg_count[questionElement.text] = 1
    return agg_count

#Input an answer tree and multiple-choices question ID and return statistics about it.
def compute_multitextaggscore(answerTree,questionID):
    agg_count = {}
    for response in answerTree.findall('Response'):
        questionElement = response.find(questionID)
        if questionElement == None:
            continue
        elif questionElement.text == None:
            continue
        else:
            answers = questionElement.text.split(',')
            for answer in answers:
                if answer in agg_count.keys():
                    agg_count[answer] += 1
                else:
                    agg_count[answer] = 1
    return agg_count



#Input an answer tree and return statistics the meaning of NULLs.
def meaning_of_null_agg_score(answerTree):

    def powerset(iterable):
        result = {}
        i = 0
        s = list(iterable)
        for subset in sorted(set(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))) :
            result[i] = set(subset)
            i+= 1
        return result

    options = {
    'It denotes a non-applicable field',
    'The value does not exist',
    'The value exists and could be anything',
    'The value exists and is equal to an unknown constant',
    'Nothing is known about the value',
    'The value exists and is equal to a known constant (for example 0 or empty string)',
    'Other [Please specify]',
    'The data is dirty',
    'There is a bug'}

    optionsDict = powerset(options)
    questionID = 'QID64'
    agg_count = {}
    for i in sorted(optionsDict):
        agg_count[i] = 0

    for response in answerTree.findall('Response'):
        questionElement = response.find(questionID)
        if questionElement == None:
            continue
        elif questionElement.text == None:
            continue
        else:
            answers = set(questionElement.text.split(','))
            for i in sorted(optionsDict):
                if answers == optionsDict[i]:
                    agg_count[i] += 1
    return agg_count


#The encoding of the SQL answer for each generic query question.
sql_answers = {
'QID110':{'c1':1,'c2':0,'c3':0,'c4':0,'c5':0},
'QID179':{'c1':0,'c2':0,'c3':1,'c4':0,'c5':0},
'QID206':{'o1':0,'o2':0,'o3':0,'o4':0,'o5':0,'o6':0},
'QID209':{'o1':0,'o2':0,'o3':0,'o4':0,'o5':1,'o6':1},
'QID20':{'c1':0,'c2':0,'c3':1,'c4':1,'c5':1,'NULL':0},
'QID19':{'c1':0,'c2':0,'c3':1,'c4':1,'c5':1,'NULL':0},
'QID21':{'c1':0,'c2':0,'c3':1,'c4':0,'c5':0,'NULL':0},
'QID226':{'c1':0,'c2':0,'c3':1,'c4':0,'c5':0,'NULL':0},
'QID86':{'c1':0,'c2':0,'c3':2,'c4':0,'c5':0,'NULL':2}
}

#Input an answer tree and generic question ID and output statistics about it.
def generic_queries_stat(answerTree,questionID):
    equal = 0
    subset = 0
    superset = 0
    neither = 0
    for response in answerTree.findall('Response'):
        equal_temp = True
        subset_temp = False
        superset_temp = False
        respondentTable = {}
        questionElement = response.find(questionID)
        if questionElement == None:
            continue
        for row in list(questionElement):
            if row.tag == 'Other' or row.tag == 'sqlasked' or row.tag == 'natLanguage':
                continue
            respondentTable[row.tag] = ast.literal_eval(row.text)
        for key in sql_answers[questionID]:
            if sql_answers[questionID][key] != (respondentTable[key])[-1]:
                equal_temp = False
            if sql_answers[questionID][key] > (respondentTable[key])[-1]:
                subset_temp = True
            if sql_answers[questionID][key] < (respondentTable[key])[-1]:
                superset_temp = True
        if equal_temp:
            equal += 1
        elif superset_temp and subset_temp:
            neither += 1
        elif superset_temp:
            superset += 1
        elif subset_temp:
            subset+=1
    total = equal+neither+superset+subset
    return(equal*100.0/total,neither*100.0/total,superset*100.0/total,subset*100.0/total)


#The table id of the SQL answer for each value inventing query question.
sql_answers_agg = {
'QID234':'QID234-3-track',
'QID236':'QID236-1-track',
'QID93':'QID93-3-track',
'QID119':'QID119-1-track',
'QID96':'QID96-3-track',
'QID97':'QID97-3-track',
'QID98':'QID98-1-track',
'QID106':'QID106-1-track',
'QID200':'QID200-9-track',
'QID229':'QID229-1-track',
'QID109':'QID109-3-track',
'QID118':'QID118-3-track',
'QID113':'QID113-3-track',
}


#Input an answer tree and a value inventing question ID and output the average SQL satisfaction score.
def agg_stat(answerTree,questionID):
    agg_score = 0
    agg_count = 0
    for response in answerTree.findall('Response'):
        questionElement = response.find(questionID)
        if questionElement == None:
            continue
        sqlElement = questionElement.find(sql_answers_agg[questionID])
        sqlrank = ast.literal_eval(sqlElement.text)
        if sqlrank[-1] == None:
            continue
        agg_count += 1
        agg_score += sqlrank[-1]
    return agg_score/agg_count

#Input an answer tree and a value inventing question ID and output the list of SQL satisfaction score of respondents.
def sql_stat(answerTree,questionID):
    sql_list = []
    for response in answerTree.findall('Response'):
        questionElement = response.find(questionID)
        if questionElement == None:
            continue
        sqlElement = questionElement.find(sql_answers_agg[questionID])
        sqlrank = ast.literal_eval(sqlElement.text)
        if sqlrank[-1] == None:
            continue
        sql_list += [sqlrank[-1]]
    return sql_list


#The table id of the Custom answer for each value inventing query question.
other_answers_agg = {
'QID234':'QID234-5-track',
'QID236':'QID236-6-track',
'QID93':'QID93-6-track',
'QID119':'QID119-6-track',
'QID96':'QID96-5-track',
'QID97':'QID97-5-track',
'QID98':'QID98-7-track',
'QID106':'QID106-7-track',
'QID200':'QID200-7-track',
'QID229':'QID229-7-track',
'QID109':'QID109-9-track',
'QID118':'QID118-10-track',
'QID113':'QID113-10-track',
}

#Input an answer tree and a value inventing question ID and output the proportion of answers which not consider SQL as the best possible answer.
def get_proportion_alternative(answerTree,qid):
    total = 0.0
    worst = 0.0
    for response in answerTree.findall('Response'):
        sql = 0
        questionElement = response.find(qid)
        if questionElement == None:
            continue
        elementKey = sql_answers_agg[qid]
        tupleElement = questionElement.find(elementKey)
        if tupleElement == None:
            continue
        tupleScore = ast.literal_eval(tupleElement.text)
        if tupleScore == None:
            continue
        if tupleScore[-1] == None:
            continue
        sql = tupleScore[-1]

        for i in range(15):
            elementKey = qid+'-'+str(i)+'-track'
            #Exclude custom answers scores
            if elementKey == other_answers_agg[qid]:
                continue
            tupleElement = questionElement.find(elementKey)
            if tupleElement == None:
                continue
            if elementKey == other_answers_agg[qid]:
                continue
            tupleScore = ast.literal_eval(tupleElement.text)
            if tupleScore == None:
                continue
            if tupleScore[-1] == None:
                continue
            if sql < tupleScore[-1]:
                worst+=1.0
        total+=1.0
    return worst/total * 100





#Data-exploration and machine learning analysis.

import numpy as np
from dython.nominal import theils_u
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold


#Preparing the data: Create the matrice of demographic datapoints

def get_participants_demographics(answerTree):
    participants_demographics = []
    respondant_header = []

    academic_response = academic_filter(answerTree)
    backend_response = filter_union(question_output_filter_equal(answerTree,'QID171','Often'),question_output_filter_equal(answerTree,'QID171','Regularly'))

    i = -1
    for response in answerTree.findall('Response'):
        i+= 1
        respondant_array = []
        responseID = response.find('_recordId').text

        #Check if academic and populate respondant_array accordingly
        academic = False
        for temp in academic_response.findall('Response'):
            if responseID == temp.find('_recordId').text:
                academic = True
        respondant_array += [academic]
        if i == 0:
            respondant_header += ['academic']

        #Check if backend user and populate respondant_array accordingly
        backend = False
        for temp in backend_response.findall('Response'):
            if responseID == temp.find('_recordId').text:
                backend = True
        respondant_array += [backend]
        if i == 0:
            respondant_header += ['backend']

        semantic_set = response.find('QID64')
        semantic_set = set(semantic_set.text.split(','))
        respondant_array += [1 if 'The value does not exist' in semantic_set or 'It denotes a non-applicable field' in semantic_set else 0]
        if i == 0:
            respondant_header += ['NA']
        respondant_array += [1 if 'Nothing is known about the value' in semantic_set else 0]
        if i == 0:
            respondant_header += ['NI']
        respondant_array += [1 if 'The value exists and is equal to an unknown constant' in semantic_set or 'The value exists and could be anything' in semantic_set else 0]
        if i == 0:
            respondant_header += ['EU']
        respondant_array += [1 if 'The data is dirty' in semantic_set or 'There is a bug' in semantic_set else 0]
        if i == 0:
            respondant_header += ['D']
        respondant_array += [1 if 'Other [Please specify]' in semantic_set else 0]
        if i == 0:
            respondant_header += ['O']
        respondant_array += [1 if 'The value exists and is equal to a known constant (for example 0 or empty string)' in semantic_set else 0]
        if i == 0:
            respondant_header += ['C']

        #Append the row to the matrix
        if participants_demographics == []:
            participants_demographics = np.array([respondant_array])
        else:
            participants_demographics = np.append(participants_demographics,[respondant_array],axis = 0)

    return (respondant_header,participants_demographics)



#Preparing the data: Create the matrice of sql answers datapoints

def get_participants_datapoints_sql_binary(answerTree):
    participants_datapoints_sql_binary=[]
    respondant_data_sql_binary_header = []

    academic_response = academic_filter(answerTree)
    backend_response = filter_union(question_output_filter_equal(answerTree,'QID171','Often'),question_output_filter_equal(answerTree,'QID171','Regularly'))

    i = -1
    for response in answerTree.findall('Response'):
        i+= 1
        respondant_data_sql_binary = []
        responseID = response.find('_recordId').text

        generic_queries = [('QID110',{'c1':1,'c2':0,'c3':0,'c4':0,'c5':0}),
        ('QID179',{'c1':0,'c2':0,'c3':1,'c4':0,'c5':0}),
        ('QID206',{'o1':0,'o2':0,'o3':0,'o4':0,'o5':0,'o6':0}),
        ('QID209',{'o1':0,'o2':0,'o3':0,'o4':0,'o5':1,'o6':1}),
        ('QID20',{'c1':0,'c2':0,'c3':0,'c4':0,'c5':0,'NULL':0}),
        ('QID19',{'c1':0,'c2':0,'c3':1,'c4':1,'c5':1,'NULL':0}),
        ('QID86',{'c1':0,'c2':0,'c3':2,'c4':0,'c5':0,'NULL':2})]

        for qid,sql_result in generic_queries:
            equal_temp = True
            subset_temp = False
            superset_temp = False
            respondentTable = {}
            questionElement = response.find(qid)
            for row in list(questionElement):
                #print(row.text)
                if row.tag == 'Other' or row.tag == 'sqlasked' or row.tag == 'natLanguage':
                    continue
                respondentTable[row.tag] = ast.literal_eval(row.text)
            for key in sql_result:
                if sql_result[key] != (respondentTable[key])[-1]:
                    equal_temp = False
                if sql_result[key] > (respondentTable[key])[-1]:
                    subset_temp = True
                if sql_result[key] < (respondentTable[key])[-1]:
                    superset_temp = True
            if equal_temp:
                respondant_data_sql_binary += [0]
            elif superset_temp and subset_temp:
                respondant_data_sql_binary += [1]
            elif superset_temp:
                respondant_data_sql_binary += [1]
            elif subset_temp:
                respondant_data_sql_binary += [1]
            if i == 0:
                respondant_data_sql_binary_header += [qid]

        exept_queries = [('QID21',{'c1':0,'c2':0,'c3':1,'c4':0,'c5':0,'NULL':0}),
        ('QID226',{'c1':0,'c2':0,'c3':1,'c4':0,'c5':0,'NULL':0})]

        equal_temp = True
        subset_temp = False
        superset_temp = False
        respondentTable = {}

        for qid,sql_result in exept_queries:
            questionElement = response.find(qid)
            if questionElement == None:
                continue
            for row in list(questionElement):
                #print(row.text)
                if row.tag == 'Other' or row.tag == 'sqlasked' or row.tag == 'natLanguage':
                    continue
                respondentTable[row.tag] = ast.literal_eval(row.text)
            for key in sql_result:
                if sql_result[key] != (respondentTable[key])[-1]:
                    equal_temp = False
                if sql_result[key] > (respondentTable[key])[-1]:
                    subset_temp = True
                if sql_result[key] < (respondentTable[key])[-1]:
                    superset_temp = True
            if equal_temp:
                respondant_data_sql_binary += [0]
            elif superset_temp and subset_temp:
                respondant_data_sql_binary += [1]
            elif superset_temp:
                respondant_data_sql_binary += [1]
            elif subset_temp:
                respondant_data_sql_binary += [1]
            if i == 0:
                respondant_data_sql_binary_header += ['exept_queries']

        sql_answers_agg = {
        'QID234':'QID234-3-track',
        'QID236':'QID236-1-track',
        'QID93':'QID93-3-track',
        'QID119':'QID119-1-track',
        'QID96':'QID96-3-track',
        'QID97':'QID97-3-track',
        'QID98':'QID98-1-track',
        'QID106':'QID106-1-track',
        'QID200':'QID200-9-track',
        'QID229':'QID229-1-track',
        'QID109':'QID109-3-track',
        'QID118':'QID118-3-track',
        'QID113':'QID113-3-track',
        }

        for qid in sql_answers_agg.keys():
            questionElement = response.find(qid)
            elementKey = sql_answers_agg[qid]
            tupleElement = questionElement.find(elementKey)
            if tupleElement == None:
                continue
            tupleScore = ast.literal_eval(tupleElement.text)
            if tupleScore[-1] == None:
                tupleScore[-1] = 0
            sql_score = tupleScore[-1]

            sql_is_max = True

            for j in range(15):
                elementKey = qid+'-'+str(j)+'-track'
                if  elementKey == other_answers_agg[qid]:
                    continue
                if  elementKey == sql_answers_agg[qid]:
                    continue
                tupleElement = questionElement.find(elementKey)
                if tupleElement == None:
                    continue
                tupleScore = ast.literal_eval(tupleElement.text)
                if tupleScore[-1] == None:
                    tupleScore[-1] = 0
                if tupleScore[-1] > sql_score:
                    sql_is_max = False
            respondant_data_sql_binary += [0 if sql_is_max else 1]
            if i == 0:
                respondant_data_sql_binary_header += [qid]


        if participants_datapoints_sql_binary == []:
            participants_datapoints_sql_binary = np.array([respondant_data_sql_binary])
        else:
            participants_datapoints_sql_binary = np.append(participants_datapoints_sql_binary,[respondant_data_sql_binary],axis = 0)

    return (respondant_data_sql_binary_header,participants_datapoints_sql_binary)


def predict_feature(demographics,sql_datapoints,qid):

    X_dem = demographics[1]
    X_all = X_dem
    X_unique = np.array([hash(X_all[i].tobytes()) for i in range(0,X_all.shape[0])])
    #X_rand = np.random.randint(2, size= X_all.shape)

    Y_datapoint = sql_datapoints[1]
    Y_qid = sql_datapoints[0]
    feature_idx_to_predict = np.where(Y_qid == qid)

    Y_all = Y_datapoint[:, feature_idx_to_predict]

    uncertainty_coefs = theils_u(Y_all,X_unique)
    rand_uncertainty_coefs = []
    val_accs_lr_full = []
    val_accs_rf_full = []
    val_accs_nn_full = []
    val_accs_dum_full = []


    for iteration in range(0,20):
        X_rand = np.array(X_all, copy=True)
        list(map(np.random.shuffle, X_rand))
        X__rand_unique = np.array([hash(X_rand[i].tobytes()) for i in range(0,X_rand.shape[0])])
        rand_uncertainty_coefs += [theils_u(Y_all,X__rand_unique)]

        n_folds = 5
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

        val_accs_lr = []
        val_accs_rf = []
        val_accs_dum = []
        val_accs_nn = []

        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X_all, Y_all)):
            model_dum = DummyClassifier(strategy="most_frequent")
            model_rf = RandomForestClassifier()
            model_lr = LogisticRegression()
            model_nn = MLPClassifier(max_iter = 2000)

            X_train_split, Y_train_split, X_val_split, Y_val_split = X_all[train_indices], Y_all[train_indices], X_all[val_indices], Y_all[val_indices]
            model_dum.fit(X_train_split, Y_train_split)
            model_rf.fit(X_train_split, Y_train_split)
            model_lr.fit(X_train_split, Y_train_split)
            model_nn.fit(X_train_split, Y_train_split)

            val_accuracy_rf = accuracy_score(Y_val_split, model_rf.predict(X_val_split))
            val_accuracy_lr = accuracy_score(Y_val_split, model_lr.predict(X_val_split))
            val_accuracy_dum = accuracy_score(Y_val_split, model_dum.predict(X_val_split))
            val_accuracy_nn = accuracy_score(Y_val_split, model_nn.predict(X_val_split))

            val_accs_lr.append(val_accuracy_lr)
            val_accs_rf.append(val_accuracy_rf)
            val_accs_dum.append(val_accuracy_dum)
            val_accs_nn.append(val_accuracy_nn)

            val_accs_lr_full += [100*np.mean(val_accs_lr)]
            val_accs_rf_full += [100*np.mean(val_accs_rf)]
            val_accs_nn_full += [100*np.mean(val_accs_nn)]
            val_accs_dum_full += [100*np.mean(val_accs_dum)]

    return(uncertainty_coefs,np.mean(rand_uncertainty_coefs),np.mean(val_accs_lr_full),np.mean(val_accs_rf_full),np.mean(val_accs_nn_full),np.mean(val_accs_dum_full))
