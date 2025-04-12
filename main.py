import pandas as pd
import pickle
import streamlit as st
import my_funcs as fn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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

menu = ["Home", "Yêu cầu của doanh nghiệp","Các thật toán thử nghiệm", "Lựa chọn kết quả", "Kiểm tra kết quả"]
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Home': 
    st.markdown("<h1 style='text-align: center;'>Đồ Án Tốt Nghiệp<br>Data Science & Machine Learning</h1>", unsafe_allow_html=True)    
    st.markdown("<h2 style='text-align: center;font-weight: bold; color: blue'>Đề tài: Phân nhóm khách hàng</h1>", unsafe_allow_html=True)
    st.image('images/h3_1.png')
elif choice == 'Yêu cầu của doanh nghiệp':   
    st.image('images/CuaHang.png')
    st.write("")
    st.write("#### * Cửa hàng X chủ yếu bán các sản phẩm thiết  yếu cho khách hàng như rau, củ, quả, thịt, cá,  trứng, sữa, nước giải khát... Khách hàng của cửa hàng là khách hàng mua lẻ.")
    st.write("#### * Chủ cửa hàng X mong muốn có thể bán được nhiều hàng hóa hơn cũng như giới thiệu sản  phẩm đến đúng đối tượng khách hàng, chăm sóc và làm hài lòng khách hàng")
elif choice == "Các thật toán thử nghiệm":   
    st.write('### Tập Luật')    
    df_RFM_TapLuat.rename(columns={'RFM_Level': 'Cluster'}, inplace=True)
    df_RFM_TapLuat['ClusterName']=df_RFM_TapLuat['Cluster']

    rfm_agg3=fn.tinh_gia_tri_tb_RFM(df_RFM_TapLuat)
    st.write("**Tính giá trị trung bình RFM cho các cụm**")
    st.markdown(fn.format_table(rfm_agg3).to_html(), unsafe_allow_html=True)
    fn.ve_cac_bieu_do(rfm_agg3,df_RFM_TapLuat,st,'Tập luật')
# ----------------------------------------------
    st.markdown("---")
    st.write('### Thuật toán GMM')    
    # Gán nhãn cụm cho dữ liệu
    df_RFM['Cluster'] = gmm_model.predict(scaled_data)
    df_RFM['ClusterName'] = df_RFM['Cluster'].apply(lambda x: f'Cluster {x}')    

    rfm_agg=fn.tinh_gia_tri_tb_RFM(df_RFM)
    st.write("**Tính giá trị trung bình RFM cho các cụm**")
    st.markdown(fn.format_table(rfm_agg).to_html(), unsafe_allow_html=True)
    fn.ve_cac_bieu_do(rfm_agg,df_RFM,st,'GMM')
# ----------------------------------------------
    st.markdown("---")
    st.write('### Thuật toán KMeans')
    st.write("**Tính giá trị trung bình RFM cho các cụm**")
    st.markdown(fn.format_table(rfm_agg2).to_html(), unsafe_allow_html=True)  
    fn.ve_cac_bieu_do(rfm_agg2,df_now,st,'KMeans')  
elif choice == 'Lựa chọn kết quả': 
    # st.write("")
    # st.write('**Full Data**')
    # st.markdown(fn.format_table(df.head()).to_html(), unsafe_allow_html=True)

    # st.write("")
    # st.write('**RFM**')
    # st.markdown(fn.format_table(df_now.head()).to_html(), unsafe_allow_html=True)

    st.write("")
    st.write('**Tính giá trị trung bình RFM cho các cụm**')
    st.markdown(fn.format_table(rfm_agg2.head()).to_html(), unsafe_allow_html=True)

    fn.ve_cac_bieu_do(rfm_agg2,df_now,st,'KMeans')

    # Ví dụ sử dụng với top 3 sản phẩm ưa thích
    behavior_table = df_merged.groupby('ClusterName').apply(lambda group: fn.get_top_products_info(group, df_merged, top_n=3))
    behavior_table=behavior_table.droplevel(level=1)
    st.write("")
    st.write('**Top 3 sản phẩm/nhóm sản phẩm ưa thích nhất của mỗ cụm**')
    st.markdown(fn.format_table(behavior_table.head()).to_html(), unsafe_allow_html=True)    
elif choice == 'Kiểm tra kết quả':
    st.write("")
    st.write('**Danh sách khách hàng**')
    st.markdown(fn.format_table(customers.head(10)).to_html(), unsafe_allow_html=True)

    st.write("")
    st.write('**Danh sách khách hàng ngẫu nhiên**')
    st.markdown(fn.format_table(random_customers).to_html(), unsafe_allow_html=True)

    fig_scatter=fn.truc_quan_hoa_scatter(rfm_agg2,'KMeans')
    st.write("")
    st.plotly_chart(fig_scatter)

    status = st.radio("**Chọn cách nhập thông tin khách hàng:**", ("Nhập id khách hàng:", "Nhập RFM của khách hàng:"))
    st.write(f'**{status}**')
    if status=="Nhập id khách hàng:":
        fn.select_one_customers_by_id(random_customers,df_merged,st)
    else:
        fn.select_one_customers_by_RFM(df_merged,model,st)





