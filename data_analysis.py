import streamlit as st
import pandas as pd
import numpy as np
import io
from pyDOE import fracfact
from itertools import product
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ===================== 页面全局配置 =====================
st.set_page_config(
    page_title="光伏浆料综合实验数据分析平台",
    page_icon="🧪",
    layout="wide"
)
st.title("🧪 光伏浆料实验综合分析工具")
st.caption("两大模块：1.常规浆料数据统计/异常检测/配方寻优  2.正交&全因子试验设计+极差+方差分析")
st.divider()

# 全局缓存，两个模块共享表格数据
if "df_raw" not in st.session_state:
    st.session_state["df_raw"] = None
if "exp_df" not in st.session_state:
    st.session_state["exp_df"] = None

# ===================== 双主标签分割两大功能 =====================
tab_data, tab_design = st.tabs(["一、浆料日常数据处理（3σ/统计/寻优）", "二、正交/全因子试验设计与分析"])

# ======================================================
# 标签1：原7天教程的浆料数据分析全套功能
# ======================================================
with tab_data:
    st.subheader("📂 上传光伏电池实验Excel（Eta/Voc/Isc/FF）")
    upload_data = st.file_uploader("上传.xlsx/.xls实验数据表", type=["xlsx", "xls"], key="data_upload")
    if upload_data is not None:
        try:
            df = pd.read_excel(upload_data, engine="openpyxl")
            st.session_state["df_raw"] = df
            st.success("数据读取成功！")
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"读取失败：{str(e)}")

    df_main = st.session_state["df_raw"]
    if df_main is not None:
        st.divider()
        st.subheader("1. 指标基础统计（均值/标准差/CV变异系数）")
        target_cols = ["Eta(%)", "Voc(V)", "Isc(J)", "FF"]
        exist_cols = [c for c in target_cols if c in df_main.columns]
        stat_res = []
        for col in exist_cols:
            data = df_main[col].dropna()
            mean_val = data.mean()
            std_val = data.std()
            cv = std_val / mean_val if mean_val != 0 else np.nan
            stat_res.append({
                "指标": col,
                "均值": round(mean_val, 4),
                "标准差": round(std_val, 4),
                "CV变异系数": round(cv, 4)
            })
        df_stat = pd.DataFrame(stat_res)
        st.dataframe(df_stat, use_container_width=True)

        st.divider()
        st.subheader("2. 3σ准则异常值检测")
        outlier_all = []
        df_out = df_main.copy()
        for col in exist_cols:
            data = df_main[col].dropna()
            mu = data.mean()
            sigma = data.std()
            low = mu - 3 * sigma
            high = mu + 3 * sigma
            mask = (df_main[col] < low) | (df_main[col] > high)
            out_rows = df_main[mask].copy()
            if len(out_rows) > 0:
                out_rows["异常指标"] = col
                out_rows["下限3σ"] = round(low,4)
                out_rows["上限3σ"] = round(high,4)
                outlier_all.append(out_rows)
                df_out.loc[mask, f"{col}_异常标记"] = "⚠️异常"
        if len(outlier_all) > 0:
            df_outliers = pd.concat(outlier_all, ignore_index=True)
            st.warning(f"检测到 {len(df_outliers)} 条异常数据")
            st.dataframe(df_outliers, use_container_width=True)
        else:
            st.success("所有数据均在3σ正常区间，无异常值")

        st.divider()
        st.subheader("3. 配方参数遍历寻优模块")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            solid_min = st.number_input("固含量下限", value=70.0)
            solid_max = st.number_input("固含量上限", value=90.0)
            solid_step = st.number_input("固含量步长", value=2.0)
        with col_b:
            resin_min = st.number_input("树脂含量下限", value=1.0)
            resin_max = st.number_input("树脂含量上限", value=5.0)
            resin_step = st.number_input("树脂含量步长", value=0.5)
        with col_c:
            temp_min = st.number_input("烧结温度下限", value=160.0)
            temp_max = st.number_input("烧结温度上限", value=200.0)
            temp_step = st.number_input("温度步长", value=5.0)

        if st.button("开始遍历寻优", type="primary"):
            all_results = []
            best_eta = -np.inf
            best_paras = None
            # 参数网格遍历
            solid_list = np.arange(solid_min, solid_max + solid_step, solid_step)
            resin_list = np.arange(resin_min, resin_max + resin_step, resin_step)
            temp_list = np.arange(temp_min, temp_max + temp_step, temp_step)
            for s in solid_list:
                for r in resin_list:
                    for t in temp_list:
                        # 简化预测效率公式（适配浆料逻辑）
                        eta_pred = 0.82 * s - 1.2 * r - 0.012 * (180 - t)
                        all_results.append([s, r, t, round(eta_pred, 4)])
                        if eta_pred > best_eta:
                            best_eta = eta_pred
                            best_paras = [s, r, t]
            result_df = pd.DataFrame(all_results, columns=["固含量(%)", "树脂含量(%)", "烧结温度(℃)", "预测效率(%)"])
            st.success("遍历计算完成！")
            st.write(f"最优参数组合：固含量{best_paras[0]}%，树脂{best_paras[1]}%，温度{best_paras[2]}℃，预测效率{best_eta:.3f}%")
            st.dataframe(result_df, use_container_width=True)

            # 模块1专属Excel下载
            buf1 = io.BytesIO()
            with pd.ExcelWriter(buf1, engine="openpyxl") as w:
                df_out.to_excel(w, sheet_name="原始数据+异常标记", index=False)
                df_stat.to_excel(w, sheet_name="统计汇总", index=False)
                result_df.to_excel(w, sheet_name="参数寻优全部组合", index=False)
            bin1 = buf1.getvalue()
            st.download_button(
                "📥 下载浆料数据分析全套Excel",
                data=bin1,
                file_name="浆料数据统计寻优结果.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# ======================================================
# 标签2：正交/全因子试验设计、极差、方差分析（之前完整代码）
# ======================================================
with tab_design:
    st.subheader("📂 方案1：上传已有试验Excel直接分析")
    upload_exp = st.file_uploader("上传正交/全因子试验表xlsx", type=["xlsx"], key="exp_upload")
    if upload_exp is not None:
        try:
            df_upload = pd.read_excel(upload_exp, engine="openpyxl")
            st.session_state["exp_df"] = df_upload
            st.success("文件读取成功！")
            st.dataframe(df_upload, use_container_width=True)
        except Exception as e:
            st.error(f"读取失败：{str(e)}")

    st.divider()
    st.subheader("🧪 方案2：自定义因素水平，生成试验方案")
    col1, col2 = st.columns(2)
    with col1:
        factor_count = st.number_input("因素数量", min_value=2, max_value=6, value=3)
    with col2:
        level_count = st.number_input("单因素水平数", min_value=2, max_value=4, value=3)

    factor_dict = {}
    for i in range(factor_count):
        fname = st.text_input(f"第{i+1}个因素名称", value=f"因素{i+1}")
        level_text = st.text_input(f"{fname} 水平数值（英文逗号隔开）", value="10,20,30")
        level_arr = [float(v.strip()) for v in level_text.split(",")]
        factor_dict[fname] = level_arr

    tab_full, tab_orth = st.tabs(["全因子试验表", "正交试验表"])
    # 全因子生成
    with tab_full:
        if st.button("生成全因子方案", type="primary"):
            f_names = list(factor_dict.keys())
            level_sets = list(factor_dict.values())
            all_combinations = list(product(*level_sets))
            df_full = pd.DataFrame(all_combinations, columns=f_names)
            df_full["试验结果"] = np.nan
            st.session_state["exp_df"] = df_full
            st.success(f"生成 {len(df_full)} 组全因子试验")
            st.dataframe(df_full, use_container_width=True)

    # 正交表生成
    with tab_orth:
        st.warning("正交表仅支持2/3水平")
        if st.button("生成正交方案", type="primary"):
            f_names = list(factor_dict.keys())
            try:
                gen_code = " ".join([f"x{i+1}" for i in range(len(f_names))])
                mat = fracfact(gen_code)
                df_orth = pd.DataFrame()
                for idx, fn in enumerate(f_names):
                    lev_list = factor_dict[fn]
                    col_data = [lev_list[int(abs(row[idx]))] for row in mat]
                    df_orth[fn] = col_data
                df_orth["试验结果"] = np.nan
                st.session_state["exp_df"] = df_orth
                st.success(f"正交表共 {len(df_orth)} 组")
                st.dataframe(df_orth, use_container_width=True)
            except Exception as err:
                st.error(f"正交生成失败：{err}，减少因素数量重试")

    st.divider()
    st.subheader("📊 试验数据分析面板（极差+ANOVA方差）")
    df_exp = st.session_state["exp_df"]
    if df_exp is None:
        st.info("请上传Excel或生成试验表后再进行分析")
    else:
        df_edit = st.data_editor(df_exp, use_container_width=True)
        st.session_state["exp_df"] = df_edit
        df_valid = df_edit.dropna(subset=["试验结果"])

        if len(df_valid) < 3:
            st.warning("有效试验数据不足，无法计算，请补全【试验结果】列")
        else:
            if len(factor_dict) > 0:
                factor_cols = list(factor_dict.keys())
            else:
                factor_cols = list(df_valid.columns[:-1])
            target_col = "试验结果"

            # 极差分析
            st.subheader("1. 极差分析")
            range_output = []
            R_map = {}
            for fac in factor_cols:
                unique_levs = sorted(df_valid[fac].unique())
                mean_vals = []
                for lev in unique_levs:
                    mean = df_valid[df_valid[fac] == lev][target_col].mean()
                    mean_vals.append(mean)
                R = max(mean_vals) - min(mean_vals)
                range_output.append({"因素": fac, "各水平均值": mean_vals, "极差R": R})
                R_map[fac] = R
            df_range = pd.DataFrame(range_output)
            st.dataframe(df_range, use_container_width=True)
            sort_factors = sorted(R_map.items(), key=lambda x: x[1], reverse=True)
            st.write("因素影响主次（极差越大影响越强）")
            st.info(" > ".join([f"{k}(R={v:.3f})" for k, v in sort_factors]))

            # 最优参数
            opt_dir = st.radio("指标优化方向", ["越大越好（效率）", "越小越好（内阻/杂质）"])
            best_params = {}
            for fac in factor_cols:
                lev_mean_dict = {}
                for lev in df_valid[fac].unique():
                    lev_mean_dict[lev] = df_valid[df_valid[fac] == lev][target_col].mean()
                if "越大越好" in opt_dir:
                    best_lev = max(lev_mean_dict, key=lev_mean_dict.get)
                else:
                    best_lev = min(lev_mean_dict, key=lev_mean_dict.get)
                best_params[fac] = best_lev
            st.write("最优配方组合")
            st.json(best_params)

            # 方差分析ANOVA
            st.subheader("2. 方差显著性分析 ANOVA")
            try:
                formula_items = [f"C({f})" for f in factor_cols]
                formula = f"{target_col} ~ " + " + ".join(formula_items)
                model = ols(formula, data=df_valid).fit()
                anova_result = sm.stats.anova_lm(model, typ=2)
                anova_result["显著性判定"] = anova_result["PR(>F)"].apply(lambda p: "显著★(P<0.05)" if p < 0.05 else "不显著(P≥0.05)")
                st.dataframe(anova_result, use_container_width=True)
                st.caption("判定标准：P<0.05代表该因素对指标存在显著影响")
            except Exception as e:
                st.error(f"方差分析计算异常：{str(e)}，检查各因素水平数量是否充足")

            # 试验模块Excel下载
            st.divider()
            st.subheader("💾 导出全套试验分析报告Excel")
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_edit.to_excel(writer, sheet_name="原始试验数据", index=False)
                df_range.to_excel(writer, sheet_name="极差分析", index=False)
                try:
                    anova_result.to_excel(writer, sheet_name="方差ANOVA", index=True)
                except:
                    pass
            excel_bin = buffer.getvalue()
            st.download_button(
                "📥 下载完整正交/全因子分析结果.xlsx",
                data=excel_bin,
                file_name="浆料正交全因子试验报告.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )