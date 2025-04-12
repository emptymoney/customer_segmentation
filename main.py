import pandas as pd
import pickle
import streamlit as st
import my_funcs as fn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
from streamlit_option_menu import option_menu


df=pd.read_csv('df.csv')
df_RFM=pd.read_csv('df_now.csv')
df_RFM_TapLuat=pd.read_csv('df_RFM_TapLuat.csv')
df_now=df_RFM.copy()
scaled_data=pd.read_csv('scaled_data.csv')

model = pickle.load(open('customer_segmentation_model.sav', 'rb'))
gmm_model=pickle.load(open('gmm_model.pkl', 'rb'))

df_now=fn.gan_nhan_cum_cho_khach_hang(df_now,model)
rfm_agg2=fn.tinh_gia_tri_tb_RFM(df_now)
df_merged = pd.merge(df, df_now, left_on='Member_number', right_index=True, how='inner')
customers=fn.get_list_customers(df)
random_customers = customers.sample(n=3, random_state=40)

# -----------------------------------------------------------------------------------

menu = ["Giới thiệu", "Yêu cầu của doanh nghiệp","Các thuật toán thử nghiệm", "Lựa chọn kết quả","Tra cứu nhóm khách hàng"]

with st.sidebar:
    selected = option_menu("Menu chính", menu, 
        icons=['house', '1-square', '2-square','3-square','4-square'], menu_icon="cast", default_index=0)
    
if selected == "Giới thiệu":
    st.markdown("<h1 style='text-align: center;'>Đồ Án Tốt Nghiệp<br>Data Science & Machine Learning</h1>", unsafe_allow_html=True)    
    st.markdown("<h2 style='text-align: center;font-weight: bold; color: blue'>Đề tài: Phân nhóm khách hàng</h1>", unsafe_allow_html=True)
    st.image('images/h3_1.png')
elif selected=="Yêu cầu của doanh nghiệp":
    st.image('images/CuaHang.png')
    st.write("")
    st.write("#### * Cửa hàng X chủ yếu bán các sản phẩm thiết  yếu cho khách hàng như rau, củ, quả, thịt, cá,  trứng, sữa, nước giải khát... Khách hàng của cửa hàng là khách hàng mua lẻ.")
    st.write("#### * Chủ cửa hàng X mong muốn có thể bán được nhiều hàng hóa hơn cũng như giới thiệu sản  phẩm đến đúng đối tượng khách hàng, chăm sóc và làm hài lòng khách hàng")
elif selected=="Các thuật toán thử nghiệm":
    tab1, tab2, tab3 = st.tabs(["Tập Luật", "Thuật toán GMM", "Thuật toán KMeans"])

    with tab1:
        st.write("### Tập Luật chia làm 5 cụm")
        df_RFM_TapLuat.rename(columns={'RFM_Level': 'Cluster'}, inplace=True)
        df_RFM_TapLuat['ClusterName']=df_RFM_TapLuat['Cluster']

        rfm_agg3=fn.tinh_gia_tri_tb_RFM(df_RFM_TapLuat)
        st.write("**Tính giá trị trung bình RFM cho các cụm**")
        st.markdown(fn.format_table(rfm_agg3).to_html(), unsafe_allow_html=True)
        fn.ve_cac_bieu_do(rfm_agg3,df_RFM_TapLuat,st,'Tập luật')        

    with tab2:
        st.write("### GMM chia làm 8 cụm")
        df_RFM['Cluster'] = gmm_model.predict(scaled_data)
        df_RFM['ClusterName'] = df_RFM['Cluster'].apply(lambda x: f'Cluster {x}')    

        rfm_agg=fn.tinh_gia_tri_tb_RFM(df_RFM)
        st.write("**Tính giá trị trung bình RFM cho các cụm**")
        st.markdown(fn.format_table(rfm_agg).to_html(), unsafe_allow_html=True)
        fn.ve_cac_bieu_do(rfm_agg,df_RFM,st,'GMM')        

    with tab3:
        st.write("### KMeans với k=5 ,chia làm 5 cụm")
        st.write("**Tính giá trị trung bình RFM cho các cụm**")
        st.markdown(fn.format_table(rfm_agg2).to_html(), unsafe_allow_html=True)  
        fn.ve_cac_bieu_do(rfm_agg2,df_now,st,'KMeans')
elif selected=="Lựa chọn kết quả":
    st.write("")
    st.write('#### 1. Tính giá trị trung bình RFM cho các cụm')
    st.markdown(fn.format_table(rfm_agg2.head()).to_html(), unsafe_allow_html=True)

    st.write("")
    st.write('#### 2. Các biểu đồ')
    fn.ve_cac_bieu_do(rfm_agg2,df_now,st,'KMeans')

    # Ví dụ sử dụng với top 3 sản phẩm ưa thích
    behavior_table = df_merged.groupby('ClusterName').apply(lambda group: fn.get_top_products_info(group, df_merged, top_n=3))
    behavior_table=behavior_table.droplevel(level=1)
    behavior_table=behavior_table.reset_index()

    st.write("")
    st.write('#### 3. Top 3 sản phẩm/nhóm sản phẩm ưa thích nhất của mỗi cụm')
    behavior_table['Top3_Popular_Products'] = behavior_table['Top3_Popular_Products'].apply(lambda x: '<br>'.join(x.split(',')))
    behavior_table['Top_3_Popular_Category'] = behavior_table['Top_3_Popular_Category'].apply(lambda x: '<br>'.join(x.split(',')))    
    st.markdown(fn.format_table(behavior_table.head()).to_html(), unsafe_allow_html=True)    

    st.write("##### Giải thích ClusterName:")
    fn.giai_thich_ClusterName(st)
elif selected=="Tra cứu nhóm khách hàng":
    st.write('### Dự đoán phân nhóm khách hàng')

    recency_max=int(df_merged['Recency'].max()*1.5)
    frequency_max=int(df_merged['Frequency'].max()*1.5)
    monetary_max=int(df_merged['Monetary'].max()*1.5)

    # Thiết lập random_state
    random.seed(23)  # Sử dụng giá trị 42 làm random_state
    random_R = random.sample(range(recency_max), 10)
    random_F = random.sample(range(frequency_max), 10)
    random_M = random.sample(range(monetary_max), 10)

    cus_random=pd.DataFrame(list(zip(random_R,random_F,random_M)),columns=['Recency','Frequency','Monetary'])
    cus_random=fn.gan_nhan_cum_cho_khach_hang(cus_random,model,True)

    cus_random['Member_number']=cus_random.index.map(lambda x:f'KH_{x+1}')

    # st.write("")
    # st.write('**Danh sách 10 khách hàng tạo ngẫu nhiên**')    
    # st.markdown(fn.format_table(cus_random[['Recency','Frequency','Monetary','Member_number']]).to_html(), unsafe_allow_html=True)

    status = st.radio("**Chọn cách nhập thông tin khách hàng:**", ("Nhập id khách hàng:", "Nhập RFM của khách hàng:"))
    st.write(f'**{status}**')
    if status=="Nhập id khách hàng:":
        selected_cus=fn.select_one_customers_by_id(cus_random,cus_random,True,st)
    else:
        fn.select_one_customers_by_RFM(df_merged,model,st)      





#-------------------------------------------------------------
st.sidebar.subheader("📒 Nhóm thực hiện:")
st.sidebar.write("* Nguyễn Tuấn Anh")   
st.sidebar.write("* Phan Ngọc Phương Bắc") 
st.sidebar.subheader("👩‍🏫 Giảng viên:")   
st.sidebar.write("- Cô Khuất Thùy Phương")
st.sidebar.write("**📆 Ngày báo cáo: 13/04/2025**")    




