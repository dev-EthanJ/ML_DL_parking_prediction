# 필요한 라이브러리 import
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import seaborn as sns

##################################################################################
# st.markdown을 통해 전체 틀 고정
st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 90%;
        padding-top: 5rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 5rem;
    }}
    img{{
    	max-width:40%;
    	margin-bottom:40px;
    }}
</style>
""",
        unsafe_allow_html=True,
    )
##################################################################################
with st.sidebar:
    choose = option_menu("Contents", ["About", "Visualizing", "Predicting"],
                         icons=['house', 'kanban', 'person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "4!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#fafafa"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )
##################################################################################
# 파트별 컨테이너화
# 굳이 안해도 되지만, 코드 가독성이나 구조를 위해서 사이드바를 만들기 이전에 구성하였습니다.
header_container = st.container()
visualizing_container = st.container()
stats_container = st.container()
forcasting_container = st.container()
##################################################################################
# About 페이지
if choose == "About":
    with header_container:
        st.image('http://cdn.joongboo.com/news/photo/201511/1025233_955571_3323.jpeg')
        st.subheader("주차수요확인하기")
        st.write("---")
        st.write("")
        st.subheader("팀원")
        st.write("김창언: LightGBM, 자료조사,시각화")
        st.write("도형준: CNN, PPT, 시각화 ")
        st.write("장인성: 조장, Catboost")
        st.write("한혜진: XGBoost, streamlit")
        st.write("황소윤: XGBoost, streamlit")
        st.write("---")
        link = '[GitHub](https://github.com/dev-EthanJ/ML_DL_parking_prediction.git)'
        st.markdown(link, unsafe_allow_html=True)
        link = '[PPT](https://github.com/dev-EthanJ/ML_DL_parking_prediction/blob/main/%EC%A3%BC%EC%B0%A8%EC%88%98%EC%9A%94%EC%98%88%EC%B8%A1.pptx?raw=true)'
        st.markdown(link, unsafe_allow_html=True)
##################################################################################
# Visualizing 페이지
elif choose == "Visualizing":
    with visualizing_container:
        st.title("Visualizing")
        data = pd.read_csv('train.csv')
        
        def preprocessing_re(df):
                df_error =  ['C1095', 'C2051', 'C1218', 'C1894', 'C2483', 'C1502', 'C1988']
                df = df[~df['단지코드'].isin(df_error)].reset_index(drop=True)
                df.rename(columns = {'도보 10분거리 내 지하철역 수(환승노선 수 반영)':'지하철','도보 10분거리 내 버스정류장 수':'버스'},inplace=True)
                df.drop(columns=['임대보증금','임대료','자격유형','임대건물구분'],axis = 1,inplace=True)
                df.drop(columns=['단지코드'],axis = 1,inplace=True)
                df=df.dropna(axis=0)
                df = df[['지역', '공급유형','총세대수', '전용면적', '전용면적별세대수', '공가수', '지하철', '버스', '단지내주차면수', '등록차량수']]
                return df
                
        data_f = preprocessing_re(data)
        #####################################
        st.write('---')
        st.subheader('원본데이터')
        st.dataframe(data_f)
        st.write('---')
        #####################################
        def preprocessing(df):
                # 오류 단지코드가 존재하는 행들을  사전에 제거
                df_error =  ['C1095', 'C2051', 'C1218', 'C1894', 'C2483', 'C1502', 'C1988']
                #df_error =  ['C2335', 'C1327']
                df = df[~df['단지코드'].isin(df_error)].reset_index(drop=True)
                df.rename(columns = {'도보 10분거리 내 지하철역 수(환승노선 수 반영)':'지하철','도보 10분거리 내 버스정류장 수':'버스'},inplace=True)
                df.drop(columns=['임대보증금','임대료','자격유형','임대건물구분'],axis = 1,inplace=True)
                지역_비율 = (df.groupby(['지역'])['총세대수'].count())/(df.groupby(['지역'])['총세대수'].count().sum())*100
                지역_비율=지역_비율.reset_index(name='지역_비율')
                공급유형_비율 = (df.groupby(['공급유형'])['총세대수'].count())/(df.groupby(['공급유형'])['총세대수'].count().sum())*100
                공급유형_비율=공급유형_비율.reset_index(name='공급유형_비율')
                df = pd.merge(df,지역_비율, on='지역')
                df = pd.merge(df,공급유형_비율, on='공급유형')
                df.drop(columns=['지역','공급유형','단지코드'],axis = 1,inplace=True)
                df=df.dropna(axis=0)
                df = df[['총세대수', '전용면적', '전용면적별세대수', '공가수', '지하철', '버스', '단지내주차면수', '공급유형_비율',
                        '지역_비율', '등록차량수']]
                return df
                
        data = preprocessing(data)
        ####################################
        st.subheader('전처리 데이터')
        st.dataframe(data)
        st.write('---')

        #####################################

               

        
        ###########################
        
        st.subheader("기초통계")
        st.write(data.describe())
        #######################################
        fig = px.imshow(data.corr(),text_auto=True, color_continuous_scale='RdBu_r', aspect='auto')
        fig.update_layout(title='컬럼별 상관관계',xaxis_nticks=36)
        st.plotly_chart(fig)
        st.write('---')

        ############################
        
        지역_위경도 = pd.read_csv('지역_위경도.csv',encoding = 'cp949')
        df_1 = pd.read_csv('train.csv')
        df_1 = pd.merge(df_1,지역_위경도, on='지역')
        
        def preprocessing_visualize(df_1):
                # 오류 단지코드가 존재하는 행들을  사전에 제거
                df_1_error =  ['C1095', 'C2051', 'C1218', 'C1894', 'C2483', 'C1502', 'C1988']
                #df_error =  ['C2335', 'C1327']
                df_1 = df_1[~df_1['단지코드'].isin(df_1_error)].reset_index(drop=True)
                df_1.rename(columns = {'도보 10분거리 내 지하철역 수(환승노선 수 반영)':'지하철','도보 10분거리 내 버스정류장 수':'버스'},inplace=True)
                df_1.drop(columns=['임대보증금','임대료','자격유형','임대건물구분'],axis = 1,inplace=True)
                지역_비율 = (df_1.groupby(['지역'])['총세대수'].count())/(df_1.groupby(['지역'])['총세대수'].count().sum())*100
                지역_비율=지역_비율.reset_index(name='지역_비율')
                공급유형_비율 = (df_1.groupby(['공급유형'])['총세대수'].count())/(df_1.groupby(['공급유형'])['총세대수'].count().sum())*100
                공급유형_비율=공급유형_비율.reset_index(name='공급유형_비율')
                df_1 = pd.merge(df_1,지역_비율, on='지역')
                df_1 = pd.merge(df_1,공급유형_비율, on='공급유형')
                df_1.drop(columns=['공급유형','단지코드'],axis = 1,inplace=True)
                df_1=df_1.dropna(axis=0)
                df_1 = df_1[['지역','총세대수', '전용면적', '전용면적별세대수', '공가수', '지하철', '버스', '단지내주차면수', '공급유형_비율','지역_비율', '등록차량수','위도','경도']]
                return df_1
        
        df_1 = preprocessing_visualize(df_1)
        df_1 = df_1.groupby('지역').mean()
        df_1 = df_1.reset_index()
        
        fig_1 = px.scatter_mapbox(df_1, lat="위도", lon="경도", hover_name="지역", hover_data=['총세대수', '전용면적', '전용면적별세대수', '공가수', '지하철', '버스', '단지내주차면수',
       '공급유형_비율', '지역_비율'],
                        color="등록차량수",color_continuous_scale=px.colors.sequential.Jet,size=df_1["등록차량수"], size_max=20, zoom=5, height=300)
        fig_1.update_layout(mapbox_style="open-street-map")
        fig_1.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        fig_1.update_layout(title=dict(text='<b>지역별 등록차량수 분포도</b><br><sup>Number of registered vehicles by region</sup>',
                                       x=0.5,y=0.87,font=dict(family="Arial",size=25,color="#000000")
                                       ),xaxis_title=dict(text="<b>Fail Point</b>"),
                            yaxis_title="<b>Portion(%)</b>",font=dict(family="Courier New, Monospace",size=12,
                                                                      color="#000000"),showlegend=False,margin = dict(l=10, r=10, b=10))

        st.plotly_chart(fig_1)
        
        ############################
        
        st.write('---')
        
        df_2 = pd.read_csv('age_gender_info.csv')
        df_2 = df_2.groupby('지역').mean()

        fig_3 = px.bar(df_2, x = df_2.index, y = df_2.columns, )
        fig_3.update_layout({"title": {"text": "지역별 세대(성별/연령)","x": 0.5, "y": 0.9,
                                     "font": {"size": 20}},"showlegend": True,
                                              "xaxis":{"title": "지역","showticklabels": True, "dtick": 1 },
                                              "autosize":False,"width": 800,"height": 400})
        

        st.plotly_chart(fig_3)


##################################################################################
# Predicting 페이지
elif choose == "Predicting":
    with forcasting_container:
        st.title("Predicting")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["LinearRegressor", "LightGBM", "XGBoost", "Catboost","CNN"])

        #########################
        with tab1:
                st.header("LinearRegressor")
                # 첫번째 행
                r1_col1, r1_col2, r1_col3 = st.columns(3)
                총세대수1 = r1_col1.slider("총세대수", 26, 2568)
                전용면적1 = r1_col2.slider("전용면적", 14.1, 583.4)
                전용면적별세대수1 = r1_col3.slider("전용면적별세대수", 1, 1865)
                # 두번째 행
                r2_col1, r2_col2, r2_col3 = st.columns(3)
                공가수1 = r2_col1.slider("공가수",0,55)
                지하철_option1 = (0, 1, 2, 3)
                지하철1 = r2_col2.selectbox("지하철", 지하철_option1)
                버스1 = r2_col3.slider("버스", 0,20)
                # 세번째 행
                r3_col1, r3_col2, r3_col3 = st.columns(3)
                단지내주차면수1 = r3_col1.slider("단지내주차면수",13,1798)
                공급유형_비율1 = r3_col2.slider("공급유형_비율",0,60)
                지역_비율1 = r3_col3.slider("지역_비율",0,21)
                predict_button = st.button("예측")
                
                if predict_button:
                        variable1 = np.array([총세대수1, 전용면적1, 전용면적별세대수1, 공가수1, 지하철1, 버스1, 단지내주차면수1, 공급유형_비율1, 지역_비율1])
                        model1 = joblib.load('LinearRegression.pkl')
                        pred1 = model1.predict([variable1])
                        st.metric("결과: ", pred1[0])
                
                
                
                ##############
                
                
        with tab2:
                st.header("LightGBM")
                # 첫번째 행
                r1_col1, r1_col2, r1_col3 = st.columns(3)
                총세대수2 = r1_col1.slider("총세대수.", 26, 2568)
                전용면적2 = r1_col2.slider("전용면적.", 14.1, 583.4)
                전용면적별세대수2 = r1_col3.slider("전용면적별세대수.", 1, 1865)
                # 두번째 행
                r2_col1, r2_col2, r2_col3 = st.columns(3)
                공가수2 = r2_col1.slider("공가수.",0,55)
                지하철_option2 = (0, 1, 2, 3)
                지하철2 = r2_col2.selectbox("지하철.", 지하철_option2)
                버스2 = r2_col3.slider("버스.", 0,20)
                # 세번째 행
                r3_col1, r3_col2, r3_col3 = st.columns(3)
                단지내주차면수2 = r3_col1.slider("단지내주차면수.",13,1798)
                공급유형_비율2 = r3_col2.slider("공급유형_비율.",0,60)
                지역_비율2 = r3_col3.slider("지역_비율.",0,21)
                predict_button2 = st.button("lightGBM예측")
                
                if predict_button2:
                        variable2 = np.array([총세대수2, 전용면적2, 전용면적별세대수2, 공가수2, 지하철2, 버스2, 단지내주차면수2, 공급유형_비율2, 지역_비율2])
                        model2 = joblib.load('LinearRegression.pkl')
                        pred2 = model2.predict([variable2])
                        st.metric("결과: ", pred2[0])
                
                #########
        with tab3:
                st.header("XGBoost")
                # 첫번째 행
                r1_col1, r1_col2, r1_col3 = st.columns(3)
                총세대수 = r1_col1.slider("총세대수_x", 26, 2568)
                전용면적 = r1_col2.slider("전용면적_x", 14.1, 583.4)
                전용면적별세대수 = r1_col3.slider("전용면적별세대수_x", 1, 1865)
                # 두번째 행
                r2_col1, r2_col2, r2_col3 = st.columns(3)
                공가수 = r2_col1.slider("공가수_x",0,55)
                지하철_xgb_option3 = (0, 1, 2, 3)
                지하철 = r2_col2.selectbox("지하철_x", 지하철_xgb_option3)
                버스 = r2_col3.slider("버스_x", 0,20)
                # 세번째 행
                r3_col1, r3_col2, r3_col3 = st.columns(3)
                단지내주차면수 = r3_col1.slider("단지내주차면수_x",13,1798)
                공급유형_비율 = r3_col2.slider("공급유형_비율_x",0,60)
                지역_비율 = r3_col3.slider("지역_비율_x",0,21)
                predict_button3 = st.button("XGBoost예측")
                
                if predict_button3:
                        variable3 = np.array([총세대수, 전용면적, 전용면적별세대수, 공가수, 지하철, 버스, 단지내주차면수, 공급유형_비율, 지역_비율])
                        model3 = joblib.load('XGBoostingRegressor.pkl')
                        pp = pd.DataFrame([variable3])
                        pp.columns = ['총세대수', '전용면적', '전용면적별세대수', '공가수', '지하철', '버스', '단지내주차면수', '공급유형_비율', '지역_비율']
                        pred3 = model3.predict(pp)
                        st.metric("결과: ", pred3[0])
                        
                        
        with tab4:
                st.header("Catboost")
                # 첫번째 행
                r1_col1, r1_col2, r1_col3 = st.columns(3)
                총세대수_c = r1_col1.slider("총세대수_c", 26, 2568)
                전용면적_c = r1_col2.slider("전용면적_c", 14.1, 583.4)
                전용면적별세대수_c = r1_col3.slider("전용면적별세대수_c", 1, 1865)
                # 두번째 행
                r2_col1, r2_col2, r2_col3 = st.columns(3)
                공가수_c = r2_col1.slider("공가수_c",0,55)
                지하철_c_option = (0, 1, 2, 3)
                지하철_c = r2_col2.selectbox("지하철_c", 지하철_c_option)
                버스_c = r2_col3.slider("버스_c", 0,20)
                # 세번째 행
                r3_col1, r3_col2, r3_col3 = st.columns(3)
                단지내주차면수_c = r3_col1.slider("단지내주차면수_c",13,1798)
                공급유형_비율_c = r3_col2.slider("공급유형_비율_c",0,60)
                지역_비율_c = r3_col3.slider("지역_비율_c",0,21)
                
                predict_button4 = st.button("Catboost예측")
                if predict_button4:
                        variable4 = np.array([총세대수_c, 전용면적_c, 전용면적별세대수_c, 공가수_c, 지하철_c, 버스_c, 단지내주차면수_c, 공급유형_비율_c, 지역_비율_c])
                        model4 = joblib.load('Catboost_GridSearchCV_model.pkl')
                        pred4 = model4.predict([variable4])
                        st.metric("결과: ", pred4[0])
                        
                
                ########################
                
                
        


                

              
##################################################################################
