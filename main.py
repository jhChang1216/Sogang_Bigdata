import pandas as pd
import sklearn.neural_network as ann
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, ARDRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime


def preprocessing_df(df, column=0, MA_condition=0, window = 6):
    df = df.T
    df = df.iloc[1:, column]
    index_list = list()
    item_list = list()
    try:
        df.index = pd.to_datetime(df.index)
    except: #환율데이터는 추가적인 전처리 과정 필요
        for idx in df.index:
            year = str(idx[0:4])
            month = str(idx[4:6])
            tmp = pd.to_datetime(year+'-'+month)
            index_list.append(tmp)
        df.index = index_list
        for item in df:
            try:
                tmp = item.replace(',', '')
                tmp = float(tmp)
                item_list.append(tmp)
            except:
                item_list.append(item)
        tmp = pd.DataFrame(item_list)
        tmp.index = df.index
        df = tmp
    if MA_condition:
        df = df.rolling(window=window).mean()  # 이동평균 해볼까??
    return df

def sorting_index(df_list):
    tmp = df_list[0]
    for i, item in enumerate(df_list):
        if(item.index[0] > tmp.index[0]):
            tmp = item
    return tmp

def sorting_index_min(df_list):
    tmp = df_list[0]
    for i, item in enumerate(df_list):
        if(item.index[0] < tmp.index[0]):
            tmp = item
    return tmp

def replace_NAN_to_column_mean(df, column):
    df[column] = df[column].fillna(method = 'ffill')
    df[column] = df[column].fillna(0)
    return df

def load_data():

   #path = 'C:/Users/wnsk1/OneDrive/바탕 화면/pythonProject1/datasets' #집 PC
    path = 'C:/projects/pythonProject2/Sogang_Bigdata/datasets' #숙소 PC
    df_list = []

    oil = pd.read_excel(path+'/주유소_평균_판매가격_20221109032912.xlsx')

    gasoline = preprocessing_df(oil, column=0)
    #deisel = preprocessing_df(oil, column=3)

    df_list.append(gasoline)

    interest = pd.read_excel(path+'/예금은행_대출금리_신규취급액_기준__20221109042933.xlsx')
    interest = preprocessing_df(interest)
    df_list.append(interest)

    m_currency = pd.read_excel(path+'/본원통화_구성내역_평잔__원계열__20221109045023.xlsx')
    m_currency = preprocessing_df(m_currency)
    df_list.append(m_currency)

    current_account = pd.read_excel(path+'/국제수지_20221109043720.xlsx')
    current_account = preprocessing_df(current_account)
    df_list.append(current_account)

    PPI = pd.read_excel(path+'/국내공급물가지수_20221109033416.xlsx')
    PPI = preprocessing_df(PPI, 2)
    df_list.append(PPI)

    spot_rate = pd.read_excel(path+'/환율통계.xlsx')
    spot_rate_dollor = preprocessing_df(spot_rate, 0)
    df_list.append(spot_rate_dollor)

    CLI = pd.read_excel(path+'/경기종합지수_2015100__구성지표_시계열__10차__20221113132427.xlsx')
    CLI = preprocessing_df(CLI, 1)
    df_list.append(CLI)

    house = pd.read_excel(path+'/행정구역별_아파트거래현황_20221113140133.xlsx')
    house = preprocessing_df(house)
    df_list.append(house)

    house_price = pd.read_excel(path+'/아파트_규모별_매매_실거래가격지수_20221126054706.xlsx')
    house_price = preprocessing_df(house_price)
    df_list.append(house_price)

    kospi = pd.read_excel(path+'/코스피_200_지수_20221126055840.xlsx')
    kospi = preprocessing_df(kospi)
    df_list.append(kospi)

    deposit_rate = pd.read_excel(path+'/예금은행_수신금리_신규취급액_기준__20221126060504.xlsx')
    deposit_rate = preprocessing_df(deposit_rate)
    df_list.append(deposit_rate)

    gold = pd.read_csv(path+'/금 선물 내역.csv')
    gold_index = gold['날짜']
    gold = gold['종가']
    tmp_index = list()
    for item in gold_index:
        item = item.replace(' ','')
        tmp_index.append(datetime.strptime(item, '%Y-%m-%d'))
    tmp_list = list()
    for item in gold:
        tmp_list.append(float(item.replace(',','')))
    gold = pd.DataFrame(tmp_list)
    gold.index = tmp_index
    gold = gold.loc[::-1]
    new_gold = list()
    new_gold_index = list()
    for i in range(len(tmp_index)):
        if gold.index[i].timetuple().tm_mday == 1:
            new_gold.append(gold.iloc[i,0])
            new_gold_index.append(gold.index[i])
    gold = pd.DataFrame(new_gold)
    gold.index = new_gold_index
    df_list.append(gold)

    population = pd.read_excel(path+'/대한민국인구조사.xls')
    population = preprocessing_df(population)
    population = population.fillna(method='ffill')
    df_list.append(population)

    sin_risk = pd.read_excel(path+'/신용위험.xlsx')
    sin_risk = preprocessing_df(sin_risk)
    df_list.append(sin_risk)

    CPI = pd.read_excel(path+'/소비자물가지수_2020100__20221109032835.xlsx')
    CPI = preprocessing_df(CPI, 0)
    df_list.append(CPI)

    standard_time = sorting_index(df_list).index[0]
    df_list = list(map(lambda x: x[standard_time:], df_list))
    df_list[6].index = df_list[0].index
    #경기종합지수 인덱스가 좀 이상하니까 바꿔주자

    return df_list

def scaling_data(df, target):
    tmp_df = df
    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(df)
    tmp = min_max_scaler.transform(df)
    df = pd.DataFrame(tmp, columns=df.columns, index=list(df.index.values))
    #df[target] = tmp_df[target] 그냥 target도 같이 스케일링 해버리자
    return df

def seperate_train_test(df, target, base, predict_gap):
    #데이터 분리
    #index 166 = 2019-12-01
    #print(df.index.values[base])

    #min_index_next = sorting_index_min(df_list).index[predict_gap]
    train_end = df.index.values[base]
    train_end_y = df.index.values[base+predict_gap]

    X_train = df.loc[:train_end, df.columns != target]
    X_test = df.loc[train_end:, df.columns != target]
    #X_train = df.loc[:train_end]
    #X_test = df.loc[train_end:]
    y_train = df.loc[df.index.values[predict_gap]:train_end_y, target]
    y_test = df.loc[train_end_y:, target]

    return X_train, X_test, y_train, y_test

def drop_column(X_train, X_test, column_list):
    drop_list = column_list
    X_train = X_train.drop(drop_list, axis=1)
    X_test = X_test.drop(drop_list, axis=1)
    return X_train, X_test

def step_backward(X_train, X_test, y_train):
    BIC_list = list()
    while(True):
        x = sm.add_constant(X_train)
        model = sm.OLS(y_train.values, x).fit()

        #print("BIC : ", model.bic)
        BIC_list.append(model.bic)
        model.pvalues = model.pvalues.drop('const')
        max_pval_idx = model.pvalues.idxmax()
        if model.pvalues[max_pval_idx] > 5.0e-02:
            print(max_pval_idx,' dropped.')
            X_train, X_test = drop_column(X_train, X_test, [max_pval_idx])
        else:
            return X_train, X_test, BIC_list

def predict_data(model_list, X_train, y_train, X_test, predict_gap):
    y_hat_list = list(0 for i in range(len(model_list)))

    for i, model in enumerate(model_list):
        model.fit(X_train, y_train)
        temp = model.predict(X_test)
        temp = pd.Series(temp)
        y_hat_list[i] = temp[:-predict_gap]
        y_hat_list[i].index = y_test.index[:]

    return y_hat_list

def display_prediction(model_list, model_dict, X_train, X_test, target, y_test, y_hat_list, predict_gap):
    inv_model_dict = {v: k for k, v in model_dict.items()}

    for i in range(len(model_list)):
        plt.plot(y_hat_list[i], label=inv_model_dict[model_list[i]])
    plt.plot(y_test, label='real data')
    plt.plot(df.loc[X_train.index[0]:X_test.index[0], target])
    plt.legend()
    plt.xticks(rotation = 45)
    plt.title(f'{target} prediction using data from {predict_gap} month ago')
    plt.xlabel('year-month')
    plt.ylabel(f'{target}')
    plt.grid()
    plt.show()

def display_BIC(BIC_list):
    plt.scatter(range(len(BIC_list)), BIC_list)
    plt.plot(BIC_list)
    plt.title("BIC change by step backward")
    plt.xlabel("dropped columns number")
    plt.ylabel("BIC")
    plt.grid()
    plt.show()

if __name__ == '__main__':

    df_list = load_data()
    df = pd.concat([df_list[0], df_list[1], df_list[2], df_list[3], df_list[4], df_list[5], df_list[6], df_list[7], df_list[8], df_list[9], df_list[10], df_list[11], df_list[12], df_list[13], df_list[14]], axis=1)
    df.columns = ['gasoline', 'loan_rate', 'currency_vol', 'current_account', 'PPI', 'Won/dollar_rate', 'CLI', 'real_estate(volume)', 'real_estate(price)', 'kospi200', 'deposit_rate', 'gold', 'population','dept','CPI']
    print(df)

    for item in df.columns:
        df = replace_NAN_to_column_mean(df, item)
    print(df)

    target = 'Won/dollar_rate'
    # 예측하고자 하는 컬럼을 지정(부동산, 수신금리, 달러환율, 코스피, 금)

    df = scaling_data(df, target)
    #target을 제외한 나머지 컬럼(독립변수)들을 min max 스케일링한다.
    print(df)

    # tmp_df = df
    # df = df.pct_change(periods=predict_gap).dropna()
    # #데이터들을 이전 predict_gap 개월만큼의 변동율(수익율)로 변경
    # df = df.replace(np.inf, 0)
    # #df[target] = tmp_df[target]
    # print(df)

    Linear_MSE_list = list()
    MLP_MSE_list = list()
    RF_MSE_list = list()
    MSE_list = list([Linear_MSE_list, MLP_MSE_list, RF_MSE_list])

    predict_range = 12

    model_list = [Ridge(), ann.MLPRegressor(hidden_layer_sizes=(200, 200, 200)), RandomForestRegressor()]
    model_dict = {'Ridge': model_list[0],'MLP': model_list[1], 'Randomforest': model_list[2]}
    inv_model_dict = {v: k for k, v in model_dict.items()}

    for predict_gap in range(1,predict_range+1):
        X_train, X_test, y_train, y_test = seperate_train_test(df.iloc[:int(len(df) * 0.7)], target,int(len(df) * 0.7 * 0.8), predict_gap)
        # 검정용
        X_train, X_test, y_train, y_test = seperate_train_test(df, target,int(len(df) * 0.7), predict_gap)
        # 시험용

        x = sm.add_constant(X_train)
        model = sm.OLS(y_train.values, x).fit()
        print(model.summary())

        X_train, X_test, BIC_list = step_backward(X_train, X_test, y_train)
        #display_BIC(BIC_list)

        x = sm.add_constant(X_train)
        model = sm.OLS(y_train.values, x).fit()
        print(model.summary())

        # model_list = [LinearRegression(), ann.MLPRegressor(hidden_layer_sizes=(200, 200, 200)), RandomForestRegressor()]
        # model_dict = {'LinearRegression': model_list[0], 'MLPRegeressor': model_list[1], 'Randomforest': model_list[2]}

        y_hat_list = predict_data(model_list, X_train, y_train, X_test, predict_gap)
        display_prediction(model_list, model_dict, X_train, X_test, target, y_test, y_hat_list, predict_gap)

        #inv_model_dict = {v: k for k, v in model_dict.items()}
        for i, y_hat in enumerate(y_hat_list):
            MSE = mean_squared_error(y_hat, y_test)
            MSE_list[i].append(MSE)
            print(f'{inv_model_dict[model_list[i]]}s {MSE}')

    for i, MSE in enumerate(MSE_list):
        plt.plot(range(1, predict_range + 1), MSE, label = inv_model_dict[model_list[i]])
    plt.title(f'MSE according to {target} prediction period(month)')
    plt.xlabel('month')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid()
    plt.show()

    # #print(model_list[0].predict(X_test)[-predict_gap]*100)
    # #마지막 관측 데이터를 바탕으로 다음 번째의 CPI를 예측한다.