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
        st.header("주차수요 예측하기")
        st.image('http://cdn.joongboo.com/news/photo/201511/1025233_955571_3323.jpeg')
        st.subheader("Streamlit을 활용하여 ML모델을 웹으로 표현해보자!")
        st.write("---")
        st.write("")
        link = '[GitHub](https://github.com/dev-EthanJ/ML_DL_parking_prediction.git)'
        st.markdown(link, unsafe_allow_html=True)
        link = '[PPT](https://github.com/dev-EthanJ/ML_DL_parking_prediction/blob/main/%EC%A3%BC%EC%B0%A8%EC%88%98%EC%9A%94%EC%98%88%EC%B8%A1.pptx?raw=true)'
        st.markdown(link, unsafe_allow_html=True)
        st.write("Visualizing: 데이터 상관관계를 그래프로 확인해보세요.")
        st.write("Predicting: 변수를 조정하여 주차수요를 예측해보세요.")
##################################################################################
# Visualizing 페이지
elif choose == "Visualizing":
    with visualizing_container:
        st.title("Visualizing")
        data = pd.read_csv('train.csv')

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
        
        st.subheader("기초통계")
        st.write(data.describe())
        st.write('---')
        ###########################
        
        st.subheader("컬럼정보")
               
        st.subheader("Plotly를 이용한 Heatmap")
        fig = px.imshow(data.corr(),text_auto=True, color_continuous_scale='RdBu_r', aspect='auto')
        st.plotly_chart(fig)
        
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
        
        fig_1 = px.scatter_mapbox(df_1, lat="위도", lon="경도", hover_name="지역", hover_data=['총세대수', '전용면적', '전용면적별세대수', '공가수', '지하철', '버스', '단지내주차면수','공급유형_비율', '지역_비율'],color="등록차량수",color_continuous_scale=px.colors.sequential.Jet,size=df_1["등록차량수"], size_max=20, zoom=5, height=300)
        fig_1.update_layout(mapbox_style="open-street-map")
        fig_1.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

        st.plotly_chart(fig_1)




##################################################################################
