import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

st.title("Sol de Janeiro TikTok Sentiment Dashboard")

st.header("Welcome to the Capstone Project Dashboard!")

st.markdown("This dashboard provides insights and visualizations for the Capstone Project.")


#  Load datasets
try:
    df_merged = pd.read_csv('merged_df.csv')
    df_sentiment = pd.read_csv('final_sentiment_analysis_results.csv')
except FileNotFoundError:
    st.error("One or more required files (merged_df.csv, final_sentiment_analysis_results.csv) are missing.")
    st.stop()

#  Merge the sentiment results into the main EDA dataset using video_web_url as key
merged_df_all = pd.merge(
    df_merged,
    df_sentiment[['video_web_url', 'text_sentiment', 'audio_sentiment', 'video_sentiment', 'ocr_sentiment', 'overall_sentiment']],
    on='video_web_url',
    how='inner'
)

#  Time preprocessing
merged_df_all['create_time'] = pd.to_datetime(merged_df_all['create_time_iso_y'], errors='coerce')
merged_df_all['month'] = merged_df_all['create_time'].dt.to_period('M').astype(str)
merged_df_all['day_of_week'] = merged_df_all['create_time'].dt.day_name()

#  Generate "sentiment_consistency" classification column
def classify_sentiment_row(row):
    sentiments = [
        row.get('text_sentiment', 'None'),
        row.get('audio_sentiment', 'None'),
        row.get('video_sentiment', 'None'),
        row.get('ocr_sentiment', 'None')
    ]
    if all(s == 'Positive' for s in sentiments):
        return 'All Positive'
    elif all(s == 'Negative' for s in sentiments):
        return 'All Negative'
    elif all(s == 'Neutral' for s in sentiments):
        return 'All Neutral'
    else:
        return 'Mixed'

merged_df_all['sentiment_consistency'] = merged_df_all.apply(classify_sentiment_row, axis=1)

# Optional: drop rows with incomplete sentiment info if needed
# merged_df_all = merged_df_all.dropna(subset=['text_sentiment', 'audio_sentiment', 'video_sentiment', 'ocr_sentiment'])




# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "👥 Customer Segmentation",
    "📢 Marketing Optimization",
    "🤝 Influencer Strategy",
    "🧴 Product Feedback"
])

# Customer Segmentation Tab
with tab1:
    st.header("Refined Customer Segmentation Insights")


    # Reload the dataset
    try:
        df = pd.read_csv("filtered_final_df.csv")
        
    except FileNotFoundError:
        st.error("CSV file not found.")
        st.stop()

    # Combine relevant text columns
    df['text_blob'] = df[['comment_text', 'video_text', 'combined_product_tags']].fillna('').apply(
        lambda row: ' '.join(str(val) for val in row), axis=1
    )
    df['text_blob'] = df['text_blob'].astype(str).str.lower()

    # Define refined audience segments
    refined_seeds = {
        "moms": ["mom", "mother", "mama", "mommy"],
        "teens": ["teen", "teenager", "highschool", "preppy", "softgirlera"],
        "young_adults": [
            "slay", "it girl", "that girl", "bff", "roommate", "my bf", "my gf",
            "frat", "sorority", "get ready", "grwm", "vibe", "trendy", "chic"
        ],
        "men": ["man", "boy", "guy", "husband"],
        "women": ["girl", "woman", "wife", "lady"],
        "hydration_seekers": ["moisturizing", "hydrated", "dewy", "soft"],
        "sensitive_skin_users": ["itchy", "rash", "eczema", "allergic", "sensitive"],
        "glow_lovers": ["glow", "shimmer", "radiance", "bronzed"],
        "fragrance_fans": ["scent", "smells", "fragrance", "perfume", "mist"],
        "professionals": ["nurse", "doctor", "dermatologist", "creator", "reviewer", "influencer"],
        "pregnancy_safe": ["pregnancy", "pregnant", "baby bump", "pregnancy safe", "maternity"],
        "dark_skinned": ["dark skin", "brown skin", "deep tone", "melanin", "for darker skin"],

    }

    def match_segment(text):
        for segment, keywords in refined_seeds.items():
            for kw in keywords:
                if kw in text:
                    return segment
        return None

    df['refined_segment'] = df['text_blob'].apply(match_segment)
    segmented_df = df[df['refined_segment'].notna()]

    if segmented_df.empty:
        st.warning("No customer segments detected.")
    else:
        import circlify
        import matplotlib.pyplot as plt
        import numpy as np

        # Prepare counts
        segment_counts = segmented_df['refined_segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        segment_counts['NormalizedSize'] = segment_counts['Count'].apply(lambda x: np.log(x + 1))


        circles = circlify.circlify(
            segment_counts['NormalizedSize'].tolist(),  # ✅ FIXED: use NormalizedSize
            show_enclosure=False,
            target_enclosure=circlify.Circle(x=0, y=0, r=1)
        )


        # Define color mapping by group
        segment_colors = {
            # Product-performance & skin needs
            "fragrance_fans": "#8dd3c7",
            "glow_lovers": "#8dd3c7",
            "hydration_seekers": "#8dd3c7",
            "sensitive_skin_users": "#8dd3c7",
            "pregnancy_safe": "#fb8072",
            "dark_skinned": "#8dd3c7",

            # Identity & lifestyle
            "moms": "#fb8072",
            "teens": "#fdb462",
            "young_adults": "#fdb462",
            "professionals": "#fdb462",
            "men": "#80b1d3",
            "women": "#80b1d3"
        }


        # Generate circle layout
        circles = circlify.circlify(
            segment_counts['NormalizedSize'].tolist(),
            show_enclosure=False,
            target_enclosure=circlify.Circle(x=0, y=0, r=1)
        )

        # Plot layout
        # Plot layout with adjusted limits
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis('off')
        ax.set_title("Customer Segments", fontsize=16)

        # Dynamically scale plot bounds
        all_x = [circle.x for circle in circles]
        all_y = [circle.y for circle in circles]
        all_r = [circle.r for circle in circles]

        x_min = min([x - r for x, r in zip(all_x, all_r)])
        x_max = max([x + r for x, r in zip(all_x, all_r)])
        y_min = min([y - r for y, r in zip(all_y, all_r)])
        y_max = max([y + r for y, r in zip(all_y, all_r)])

        ax.set_xlim(x_min - 0.05, x_max + 0.05)
        ax.set_ylim(y_min - 0.05, y_max + 0.05)

        # Draw circles with labels
        for circle, row in zip(circles, segment_counts.itertuples()):
            x, y, r = circle.x, circle.y, circle.r
            label = row.Segment
            count = row.Count
            color = segment_colors.get(label, '#d9d9d9')
            ax.add_patch(plt.Circle((x, y), r, alpha=0.8, linewidth=2, color=color))
            ax.text(x, y, f"{label}\n({count})", ha='center', va='center', fontsize=9)

        st.pyplot(fig)

        st.markdown("""
        The above graph illustrates the distribution of customer segments identified through TikTok content.
        
        1. Men and women contribute almost equally, making a strong case for inclusive, gender-neutral content strategies.

        2. Young adults are highly engaged, often expressing themselves through slang and lifestyle cues — ideal for trendy, TikTok-native messaging.

        3. Product performance still dominates — segments like glow and fragrance continue to outweigh demographic tags, reinforcing the importance of benefit-led storytelling.
        """)

                # --- SENTIMENT STACKED BAR CHART ---

                # --- SENTIMENT STACKED BAR CHART WITH LABELS ---

        st.subheader("I. Sentiment Distribution by Segment")

        # Filter only rows with sentiment
        sentiment_df = segmented_df[segmented_df['sentiment_from_video'].notna()]

        # Group and count
        sentiment_counts = sentiment_df.groupby(['refined_segment', 'sentiment_from_video']).size().unstack(fill_value=0)

        # Normalize to percentage
        sentiment_percent = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100

        # Ensure consistent order
        sentiment_percent = sentiment_percent[["positive", "neutral", "negative"]] if "neutral" in sentiment_percent.columns else sentiment_percent[["positive", "negative"]]

        # Plot
        fig2, ax2 = plt.subplots(figsize=(11, 6))
        bottom = np.zeros(len(sentiment_percent))

        colors = {"positive": "#fdb462", "neutral": "#fb8072", "negative": "#d9d9d9"}

        for sentiment in sentiment_percent.columns:
            values = sentiment_percent[sentiment].values
            bars = ax2.bar(sentiment_percent.index, values, bottom=bottom, label=sentiment, color=colors.get(sentiment, "#cccccc"))

            # Annotate bars
            for bar, percent in zip(bars, values):
                if percent > 3:  # only show if it's visually large enough
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f'{percent:.0f}%',
                        ha='center', va='center', fontsize=8, color='black'
                    )
            bottom += values

        ax2.set_ylabel("Percentage")
        ax2.set_xlabel("Customer Segment")
        ax2.set_title("Sentiment Split per Segment (from video)")
        ax2.legend(title="Sentiment", loc="upper right")
        ax2.tick_params(axis='x', rotation=90)
        ax2.set_ylim(0, 100)

        st.pyplot(fig2)

        st.markdown("""
        This chart illustrates the percentage of positive, neutral, and negative sentiment (from video content) across customer segments identified through TikTok comments.

        1. Fragrance Fans, Glow Lovers, and Hydration Seekers show the highest concentration of positive sentiment, validating the emotional impact of product benefits like scent and finish.

        2. Segments like Professionals and Sensitive Skin Users exhibit more mixed sentiment, signaling critical or unmet expectations — a potential area for education or reformulation.

        3. Young Adults and Teens lean heavily neutral/negative, suggesting their feedback is more nuanced or cautious — possibly due to price sensitivity, social proof, or trending comparisons.
        """)

                # --- VIRALITY INDEX LOLLIPOP CHART WITH POST COUNT ---

        st.subheader("II. Virality Index by Customer Segment (Filtered by Volume)")

        # Filter rows with virality and segment
        virality_df = segmented_df[segmented_df['virality_index'].notna()]

        # Group: count + average virality
        virality_stats = virality_df.groupby('refined_segment').agg(
            post_count=('Id', 'count'),
            avg_virality=('virality_index', 'mean')
        )

        # Filter for segments with at least 5 posts
        virality_stats = virality_stats[virality_stats['post_count'] >= 5]
        virality_stats = virality_stats.sort_values(by='avg_virality')

        # Plot lollipop
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.hlines(y=virality_stats.index, xmin=0, xmax=virality_stats['avg_virality'], color='gray', alpha=0.7, linewidth=2)
        ax3.plot(virality_stats['avg_virality'], virality_stats.index, "o", color='#ff7f0e')

        # Annotate values
        for i, (segment, row) in enumerate(virality_stats.iterrows()):
            ax3.text(row['avg_virality'] + 0.05, i, f"{row['avg_virality']:.2f} ({row['post_count']} posts)", va='center', fontsize=8)

        ax3.set_xlabel("Average Virality Index")
        ax3.set_title("Which Segments Drive the Most Viral Engagement?")
        ax3.grid(axis='x', linestyle='--', alpha=0.5)

        st.pyplot(fig3)

        st.markdown("""
        This chart illustrates the average virality index across different customer segments, highlighting which groups generate the most impactful and shareable content related to Sol de Janeiro. The virality index measures how widely a video spreads, considering factors like views, shares, and engagement, with higher scores indicating greater reach.

        1. Moms generate the most viral content despite having fewer posts — their relatability and authenticity make them ideal for high-impact, shareable campaigns.

        2. Young adults and hydration seekers show strong virality, driven by lifestyle trends and benefit-oriented routines — a sweet spot for TikTok-native messaging.

        3. Fragrance fans, while large in volume, underperform in virality, suggesting potential fatigue or saturation. Refreshing content formats or narratives could reignite engagement.

        4. Lower-virality segments like teens, professionals, and sensitive skin users may need more tailored messaging or educational content to boost emotional and viral resonance.
        """)

                # --- TOP HASHTAGS PER SEGMENT ---

                # --- INTERACTIVE WORD CLOUD BY SEGMENT ---

        st.subheader("III. Top Hashtags by Segment")

        # Filter tag data
        tag_df = segmented_df.copy()
        tag_df = tag_df[tag_df['combined_product_tags'].notna()]
        tag_df['tags'] = tag_df['combined_product_tags'].str.lower().str.split('|')
        tag_df = tag_df.explode('tags')

        # User selects a segment
        selected_segment = st.selectbox("Select a segment to view tag focus", tag_df['refined_segment'].unique())

        # Compute counts
        segment_tags = tag_df[tag_df['refined_segment'] == selected_segment]['tags'].value_counts()
        all_tags = tag_df['tags'].value_counts()

        # Build word frequency dict
        word_freq = {}
        for tag in all_tags.index:
            if tag in segment_tags.index[:5]:
                word_freq[tag] = segment_tags[tag]  # normal weight
            else:
                word_freq[tag] = segment_tags.get(tag, 1) * 0.3  # gray-scaled

        # Generate word cloud
        from wordcloud import WordCloud
        wc = WordCloud(width=1000, height=400, background_color='white', colormap='Dark2')
        wc.generate_from_frequencies(word_freq)

        fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)

        st.markdown(f"""
        The above word cloud visualizes the most frequently used hashtags and product tags by the fragrance_fans segment. Top 5 tags are highlighted in vibrant color, while others are de-emphasized in gray to focus attention on what matters most.
        
        1. Sol de Janeiro and its core perfume lines dominate the conversation, confirming strong brand recognition and product association within this segment.

        2. Tags like fyp, unboxing, and perfumeTikTok suggest that content is being positioned to trend and perform well in TikTok’s algorithm-driven feed.

        3. High usage of product-specific hashtags (e.g., cheirosa62, bodymist, perfume) shows that users are organically linking their experiences to Sol de Janeiro's fragrance SKUs.

        4. The strong presence of emotional and sensorial words (e.g., smellgood, foryou) highlights fragrance fans’ emphasis on how products feel and are perceived, a key storytelling hook for campaign creatives.
        """)

        # --- STACKED AREA CHART: SEGMENT VOLUME OVER TIME ---

        st.subheader("IV. Segment Volume Over Time")

        # Prepare time data
        time_df = segmented_df.copy()
        time_df['create_time_iso'] = pd.to_datetime(time_df['create_time_iso'], errors='coerce')
        time_df['month'] = time_df['create_time_iso'].dt.to_period('M').astype(str)

        # Group by month and segment
        volume_by_month = (
            time_df.groupby(['month', 'refined_segment'])
            .size()
            .reset_index(name='count')
        )

        # Pivot to wide format
        pivot_area = volume_by_month.pivot(index='month', columns='refined_segment', values='count').fillna(0)
        pivot_area = pivot_area.sort_index()

        # Plot with Plotly
        import plotly.graph_objects as go
        fig_area = go.Figure()

        for segment in pivot_area.columns:
            fig_area.add_trace(
                go.Scatter(
                    x=pivot_area.index,
                    y=pivot_area[segment],
                    mode='lines',
                    stackgroup='one',
                    name=segment
                )
            )

        fig_area.update_layout(
            title="Stacked Area: Mentions Over Time by Segment",
            xaxis_title="Month",
            yaxis_title="Mentions",
            hovermode="x unified",
            height=500,
            width=1100,  # ← wider plot
            margin=dict(t=60, l=40, r=40, b=40)
        )
        
        fig_area.update_xaxes(
            tickangle=45,
            tickfont=dict(size=10),
            nticks=20,  # adjust how many ticks show up
            showgrid=True
        )

        st.plotly_chart(fig_area)

        st.markdown("""
        The stacked area chart above visualizes how engagement from different customer segments has evolved over time, highlighting key patterns in volume growth, seasonal interest, and emerging audience behavior.
        
        1. Fragrance fans dominate volume growth, especially since early 2023, showing consistent and rising interest in scent-led content.

        2. Young adults emerged strongly in the last year, now rivaling core segments like women and glow lovers — likely driven by viral trends and GRWM formats.

        3. Hydration seekers and glow lovers show seasonal lifts, suggesting interest aligns with skincare cycles (e.g., winter hydration, summer glow).

        4. Segments like sensitive skin users and professionals remain low in volume, pointing to underrepresentation or lower organic discussion — a potential opportunity for targeted education or influencer seeding.
        """)
        
        
        
        


# Marketing Optimization Tab
with tab2:
    st.header("Marketing Optimization Insights")


    # 1. Hashtag Word Cloud + Engagement Bar Chart
    st.subheader("I. Hashtags driving the most engagement")
    st.markdown("This identifies high-performing individual tags used in campaigns, based on average engagement rate.")

    # Prepare hashtag metrics
    df = df[df['combined_product_tags'].notna()]
    df = df[df['combined_product_tags'].apply(lambda x: isinstance(x, str))]

    df['combined_product_tags'] = df['combined_product_tags'].str.split('|')
    df_exploded = df.explode('combined_product_tags')
    df_exploded['combined_product_tags'] = df_exploded['combined_product_tags'].str.lower().str.strip()

    tag_metrics = (
        df_exploded
        .groupby('combined_product_tags')['engagement_rate']
        .agg(['count', 'mean'])
        .rename(columns={'count': 'usage_count', 'mean': 'avg_engagement'})
        .sort_values('avg_engagement', ascending=False)
    )

    tag_metrics = tag_metrics.dropna().query('usage_count >= 2')

    # Word Cloud
    st.markdown("**Tags by Avg Engagement**")
    wordcloud = WordCloud(width=1000, height=500, background_color='white')
    wordcloud.generate_from_frequencies(tag_metrics['avg_engagement'].to_dict())
    fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)

    # Bar Chart
    st.markdown("**Top 10 Hashtags by Avg Engagement**")
    top10_tags = tag_metrics.head(10).reset_index()
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top10_tags, x='avg_engagement', y='combined_product_tags', palette='viridis', ax=ax_bar)
    ax_bar.set_xlabel('Average Engagement Rate')
    ax_bar.set_ylabel('Hashtag')
    ax_bar.set_title('Top 10 Hashtags by Average Engagement Rate')
    st.pyplot(fig_bar)

    st.markdown("Tags like `holiday`, `mini`, `solpartner`, and `goviral` are associated with higher average engagement. These should be considered for future campaign theming.")

    st.markdown("""
                The word cloud and bar chart above highlight the hashtags most frequently used and most engaging in Sol de Janeiro TikTok content. While the word cloud emphasizes volume, the bar chart ranks tags by their average engagement rate, spotlighting what’s resonating most with audiences.
                
                1. Holiday-themed hashtags dominate both volume and engagement, signaling seasonal campaigns are especially effective for Sol de Janeiro.

                2. Tags like blowup, goviral, and lashes appear in the top 5 by engagement — indicating that users associate Sol de Janeiro content with transformational or aspirational aesthetics.

                3. Product-linked tags like 68spraysoldejanerio and mini also perform well, suggesting specific SKUs or formats drive strong interaction.

                4. The high-ranking presence of unexpected tags like ?? or ambiguous tags may reflect trending audio or viral formats — worth investigating for creative inspiration.
                """)

    # 2. Which campaigns had the most positive vs negative responses?
    # Load data
    st.header("II. Hashtags with Strong Positive and Negative Sentiment")

    try:
        df = pd.read_csv("filtered_final_df.csv")
    except FileNotFoundError:
        st.error("The file 'filtered_final_df.csv' was not found.")
        st.stop()

    # Clean and explode tags
    df = df[df['combined_product_tags'].notna()]
    df['combined_product_tags'] = df['combined_product_tags'].astype(str).str.lower().str.split('|')
    df_exploded = df.explode('combined_product_tags')
    df_exploded['combined_product_tags'] = df_exploded['combined_product_tags'].str.strip()

    # Compute sentiment
    tag_sentiment = df_exploded.groupby('combined_product_tags').agg(
        count=('Id', 'count'),
        video_pos=('sentiment_from_video', lambda x: (x == 'positive').mean()),
        video_neg=('sentiment_from_video', lambda x: (x == 'negative').mean()),
        audio_pos=('sentiment_from_audio', lambda x: (x == 'positive').mean()),
        audio_neg=('sentiment_from_audio', lambda x: (x == 'negative').mean())
    ).query('count >= 3').copy()

    tag_sentiment['net_video_sentiment'] = tag_sentiment['video_pos'] - tag_sentiment['video_neg']
    tag_sentiment['net_audio_sentiment'] = tag_sentiment['audio_pos'] - tag_sentiment['audio_neg']
    tag_sentiment['avg_net_sentiment'] = tag_sentiment[['net_video_sentiment', 'net_audio_sentiment']].mean(axis=1)

    # Select top 10 best and worst
    worst_tags = tag_sentiment.sort_values(by='avg_net_sentiment', ascending=True).head(10).reset_index()
    best_tags = tag_sentiment.sort_values(by='avg_net_sentiment', ascending=False).head(10).reset_index()

    # Plot
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

    # Top positive sentiment
    sns.barplot(
        data=best_tags,
        x='avg_net_sentiment',
        y='combined_product_tags',
        palette='Greens_d',
        ax=axes[0]
    )
    axes[0].set_title("Top 10 Tags with Most Positive Net Sentiment")
    axes[0].set_xlabel("Avg Net Sentiment (Video + Audio)")
    axes[0].set_ylabel("Hashtag")

    # Top negative sentiment
    sns.barplot(
        data=worst_tags,
        x='avg_net_sentiment',
        y='combined_product_tags',
        palette='Reds_d',
        ax=axes[1]
    )
    axes[1].set_title("Top 10 Tags with Most Negative Net Sentiment")
    axes[1].set_xlabel("Avg Net Sentiment (Video + Audio)")
    axes[1].set_ylabel("Hashtag")

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    - **Top Chart:** Tags your audience responded positively to — promote more.
    - **Bottom Chart:** Tags that drove negative sentiment — rework or retire.
    - This view blends both **video and audio sentiment** for complete signal.
    """)

    st.markdown("""
                The chart above displays the top 10 hashtags with the highest and lowest net sentiment based on both video and audio signals. The top panel highlights hashtags that resonated positively with audiences, while the bottom panel surfaces those associated with more critical or negative reactions.
                
                1. Lifestyle-driven hashtags like wishlist, dupealert, and bubble score high on positive sentiment — suggesting a strong emotional response to aspirational or affordable beauty content.

                2. Tags linked to discovery and summer perfume content (e.g., coquetteaesthetic, summerperfume) also perform well, signaling mood-based campaigns resonate with users.

                3. Negative sentiment clusters around general makeup terms like newmakeup, makeuptutorial, and makeuphacks, which may reflect fatigue with overused content formats or unmet product expectations.

                4. Hashtags like beautyinfluencer and hairgrowthtips underperform, potentially signaling skepticism or oversaturation in influencer-led or benefit-heavy messaging.
                """)
    
    # 3. Are there specific times when people engage more with campaigns?
    st.header("III. Engagement by Day of Week and Month")

    if 'engagement_rate' in df.columns and 'create_time_iso' in df.columns:
        df = df[df['create_time_iso'].notna()]
        df['create_time_iso'] = pd.to_datetime(df['create_time_iso'], errors='coerce')
        df = df[df['create_time_iso'].notna()]  # Ensure valid datetime rows

        df['weekday'] = df['create_time_iso'].dt.day_name()
        df['month'] = df['create_time_iso'].dt.strftime('%B')

        # WEEKDAY ENGAGEMENT BAR CHART
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_engagement = df.groupby('weekday')['engagement_rate'].mean().reindex(weekday_order)

        st.subheader("a) Engagement by Day of the Week")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        weekday_engagement.plot(kind='bar', color='skyblue', ax=ax1)
        ax1.set_ylabel("Average Engagement Rate")
        ax1.set_title("Average Engagement by Weekday")
        st.pyplot(fig1)

        st.markdown("""
                    The charts above show how average engagement rates for Sol de Janeiro TikTok content vary by weekday. These visualizations help identify the most effective timing windows for content release based on historical engagement trends.
                    
                    1. Thursday stands out with a significant spike in average engagement, making it the strongest day for posting high-impact content.

                    2. Weekends show the lowest engagement rates, suggesting that user attention may be lower or content is less likely to resonate on those days.

                    3. May delivers a peak in monthly engagement, potentially aligned with a major launch or seasonal push — an ideal moment to replicate campaign formats or messaging themes.

                    4. There is a disconnect between post volume and engagement in some months, reinforcing that timing and content relevance matter more than quantity alone.
                    """)
    
        # MONTHLY ENGAGEMENT COMBO CHART
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        monthly_summary = df.groupby('month').agg(
            avg_engagement=('engagement_rate', 'mean'),
            total_posts=('Id', 'count')
        ).reindex(month_order)

        monthly_summary.dropna(how='all', inplace=True)

        if not monthly_summary.empty:
            st.subheader("b) Engagement by Month of the Year")
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            monthly_summary['avg_engagement'].plot(kind='bar', ax=ax2, color='coral', label='Avg Engagement', alpha=0.7)
            ax3 = ax2.twinx()
            monthly_summary['total_posts'].plot(kind='line', ax=ax3, color='darkblue', marker='o', label='Total Posts')
            ax2.set_ylabel("Average Engagement Rate")
            ax3.set_ylabel("Total Posts")
            ax2.set_title("Engagement Trend Across Months (Bar = Engagement, Line = Post Volume)")
            ax2.legend(loc='upper left')
            ax3.legend(loc='upper right')
            st.pyplot(fig2)
        else:
            st.warning("Monthly data could not be plotted due to missing values.")
    else:
        st.warning("Required columns 'engagement_rate' and 'create_time_iso' not found in dataset.")

    st.markdown(
                """
                The chart illustrates the monthly trend of average engagement rate (bar) and post volume (line) related to Sol de Janeiro on TikTok. This dual-axis view highlights when content performs best and how it aligns with posting behavior across the year.
                
                1. May shows an exceptionally high engagement rate, suggesting a major campaign or viral moment during that month — worth analyzing further for repeatable success signals.

                2. November has the highest post volume, likely driven by holiday season buzz, but engagement rate remains modest — indicating potential content saturation.

                3. Engagement spikes don’t always align with posting volume, emphasizing the importance of quality and timing over quantity in campaign planning.
                """)
    # 4. Does promotion correlate with higher sentiment?
    st.subheader("IV. Products viewed most positively or negatively")

    # Load and prepare data
    try:
        df = pd.read_csv("filtered_final_df.csv")
    except FileNotFoundError:
        st.error("The file 'filtered_final_df.csv' was not found.")
        st.stop()

    # Define allowed product tags
    allowed_tags = [
        "perfume", "cheirosa62", "bodymist", "skincare", "bodycare", "bumbumcream", "makeup",
        "bodyspray", "soldejaineromist", "bodysplash", "bodycream", "rioradiance", "cheirosa40",
        "bodyoil", "cheirosa59", "braziliancrush", "cheirosa71", "bodybutter"
    ]

    if 'combined_product_tags' in df.columns:
        df = df[df['combined_product_tags'].notna()].copy()
        df['combined_product_tags'] = df['combined_product_tags'].astype(str).str.lower().str.split('|')
        df = df.explode('combined_product_tags')
        df['combined_product_tags'] = df['combined_product_tags'].str.strip()
        df = df[df['sentiment_from_video'].notna()]
        df = df[df['combined_product_tags'].isin(allowed_tags)]

        selected_tags = st.multiselect("Select products to visualize", sorted(allowed_tags))

        if selected_tags:
            filtered_df = df[df['combined_product_tags'].isin(selected_tags)]
            sentiment_counts = (
                filtered_df.groupby(['combined_product_tags', 'sentiment_from_video'])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=["positive", "neutral", "negative"], fill_value=0)
            )

            # Convert to percent and sort by positivity
            sentiment_percent = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100
            sentiment_percent = sentiment_percent.sort_values(by="positive", ascending=False)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            bottom = np.zeros(len(sentiment_percent))

            colors = {
                "positive": "#a1d99b",  # pastel green
                "neutral": "#f0f0f0",   # light gray
                "negative": "#fc9272"   # pastel red
            }

            for sentiment in sentiment_percent.columns:
                values = sentiment_percent[sentiment].values
                bars = ax.bar(sentiment_percent.index, values, bottom=bottom, label=sentiment, color=colors.get(sentiment))
                for i, value in enumerate(values):
                    if value > 5:
                        ax.text(i, bottom[i] + value / 2, f"{value:.0f}%", ha='center', va='center', fontsize=8)
                bottom += values

            ax.set_title("Sentiment Distribution for Selected Product Tags")
            ax.set_ylabel("Percentage")
            ax.set_xlabel("Product")
            ax.set_ylim(0, 100)
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title="Sentiment")
            st.pyplot(fig)

            st.markdown("The chart above illustrates the sentiment distribution for selected Sol de Janeiro product tags based on TikTok content. Each bar represents a product, with sections showing the percentage of positive, neutral, and negative sentiment derived from video data. The bars are sorted by the highest percentage of positive sentiment, allowing for a quick scan of how each product is perceived emotionally.")
        else:
            st.info("Please select one or more product tags to display sentiment.")
    else:
        st.warning("Column 'combined_product_tags' not found in dataset.")

    st.markdown("The chart above illustrates the sentiment distribution for selected Sol de Janeiro product tags based on TikTok content. Each bar represents a product, with sections showing the percentage of positive, neutral, and negative sentiment derived from video data. The bars are sorted by the highest percentage of positive sentiment, allowing for a quick scan of how each product is perceived emotionally.")

# Influencer Strategy Tab
with tab3:
        
    st.header("""
    Influencer Strategy Insights""")
    
    
    st.subheader("I. Emotional Delivery Patterns Across Modalities: Influencers vs. Non-Influencers")
    
    st.markdown("""
                
    The boxplot below compares sentiment scores across **Text, Audio, Video, and OCR**, using:
    - **1 = Positive**
    - **0 = Neutral**
    - **-1 = Negative**

    """)

    # Convert sentiment to numeric for plotting
    sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    melted_sentiment = merged_df_all.melt(
        id_vars=['is_influencer'],
        value_vars=['text_sentiment', 'audio_sentiment', 'video_sentiment', 'ocr_sentiment'],
        var_name='modality',
        value_name='sentiment'
    )
    melted_sentiment['sentiment_score'] = melted_sentiment['sentiment'].map(sentiment_mapping)

    # Plot the sentiment distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=melted_sentiment,
        x='modality',
        y='sentiment_score',
        hue='is_influencer',
        ax=ax
    )
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title("Sentiment Distribution by Modality: Influencers vs. Non-Influencers")
    ax.set_ylabel("Sentiment Score (-1 = Negative, 0 = Neutral, 1 = Positive)")
    ax.set_xlabel("Content Modality")
    ax.legend(title='Is Influencer')
    st.pyplot(fig)

    st.markdown("""
    **Business Insight:**

    - Both influencers and non-influencers display **widely mixed emotional signals** across all modalities — indicating that tone consistency is a platform-wide challenge rather than creator-type specific.
    - Video and OCR sentiment in particular are **highly variable**, suggesting creators may struggle to deliver clear or cohesive visual/emotional messaging through these channels.
    - This reveals an opportunity to support creators with clearer creative direction, especially for modalities prone to emotional misalignment.

    **Actionable Strategy:**

    - Provide **creative guidelines tailored to each modality** (e.g., tone-appropriate subtitles, soundtrack pairing tips, product-focused framing).
    - Use **modality-based coaching** when onboarding influencers — e.g., help strong talkers improve visual delivery, and visual creatives improve spoken clarity.
    - Encourage creators to **review their content through multiple lenses** (e.g., how it sounds vs. how it looks) to catch potential tone mismatches before posting.""")
        
    # -------------------------
    # NEW: Sentiment Consistency by Influencer Type
    # -------------------------
    st.subheader("II. Sentiment Consistency by Influencer Type")

    # Drop rows with missing consistency classification
    seg_df = merged_df_all.dropna(subset=['sentiment_consistency'])

    # Group by influencer vs non-influencer
    consistency_group = seg_df.groupby(['is_influencer', 'sentiment_consistency']).size().unstack(fill_value=0)

    # Normalize to percentage
    consistency_percent = consistency_group.div(consistency_group.sum(axis=1), axis=0) * 100

    # Plot
    fig_consistency, ax = plt.subplots(figsize=(8, 5))
    consistency_percent.T.plot(kind='bar', ax=ax, colormap='Accent')
    ax.set_ylabel("Percentage")
    ax.set_xlabel("Sentiment Consistency")
    ax.set_title("Distribution of Sentiment Consistency by Influencer Status")
    ax.legend(title="Is Influencer", labels=["Non-Influencer", "Influencer"])
    st.pyplot(fig_consistency)

    # 👉 Add clear explanation of what each category means
    st.markdown("""
    **What is Sentiment Consistency?**  
    This metric compares how well the four sentiment models (Text, Audio, Video, OCR) agree with one another for each post:

    - **"All Positive"**: All four models predicted `Positive`
    - **"All Negative"**: All four models predicted `Negative`
    - **"All Neutral"**: All four models predicted `Neutral`
    - **"Mixed"**: A mix of different sentiments (e.g., Video = Positive, Text = Neutral, etc.)

    ---

    **Business Insight:**
    - The vast majority of content—whether from influencers or regular users—shows **"Mixed" sentiment**, indicating complex emotional signals across modalities.
    - No content was flagged as “All Negative”, suggesting that strongly negative consensus across all channels is rare in the final sentiment analysis results data.

    **Actionable Strategy:**

    - **Avoid relying on a single content modality (e.g., just video, captions, or audio) when evaluating campaign sentiment.**  
    Since nearly all posts show Mixed sentiment across the four models, Sol de Janeiro should implement a **multimodal content review process** when assessing influencer or organic campaign performance.  
    ➤ For example, when reviewing TikTok content, consider both on-screen visuals (OCR), spoken tone (audio), captions (text), and viewer reactions (comments) to get a complete emotional picture.

    - **Benchmark creators based on multimodal sentiment patterns.**  
    Track and tag influencers whose posts consistently show clearer (less mixed) sentiment across all four models. These creators should be prioritized for:
    - Product launches  
    - Premium campaigns  
    - Messaging-sensitive categories (e.g., new scents, sensitive skin)  
    ➤ They are more likely to deliver emotionally cohesive, brand-aligned content that resonates consistently with target audiences.
    """)
    
    
    # --- SECTION: Irrelevant Sentiment Volume by Modality ---
    st.subheader("III. Proportion of 'Irrelevant' Sentiment by Modality")

    st.markdown("""
    This chart shows the proportion of sentiment predictions labeled as **"Irrelevant"** across all four modalities.  
    A high irrelevant rate means the content didn't convey any useful emotional tone via that channel — particularly common in OCR.

    **Modalities analyzed:**  
    - Text 
    - Audio 
    - Video  
    - OCR 
    """)

    # Recalculate irrelevant proportions
    irrelevant_counts = {
        modality: merged_df_all[modality].value_counts().get('Irrelevant', 0)
        for modality in ['text_sentiment', 'audio_sentiment', 'video_sentiment', 'ocr_sentiment']
    }
    total_counts = {
        modality: merged_df_all[modality].notna().sum()
        for modality in ['text_sentiment', 'audio_sentiment', 'video_sentiment', 'ocr_sentiment']
    }
    irrelevant_proportion = {
        modality: (irrelevant_counts[modality] / total_counts[modality]) * 100
        for modality in irrelevant_counts
    }
    irrelevant_df = pd.DataFrame({
        'modality': list(irrelevant_proportion.keys()),
        'irrelevant_percentage': list(irrelevant_proportion.values())
    })

    # Plot
    fig_irrelevant, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=irrelevant_df, x='modality', y='irrelevant_percentage', palette='Set2', ax=ax)
    ax.set_title("Proportion of 'Irrelevant' Sentiment by Modality")
    ax.set_ylabel("Irrelevant % of Total Predictions")
    ax.set_xlabel("Modality")
    ax.set_ylim(0, 100)
    st.pyplot(fig_irrelevant)

    st.markdown("""
    **Business Insight:**

    - Text, audio, and video sentiment models produce **highly interpretable emotional results**, with virtually no predictions marked as “Irrelevant.”
    - In contrast, nearly **30% of OCR-based sentiment results are labeled as “Irrelevant”**, suggesting that on-screen text often contains **non-emotional, decorative, or unstructured content**.
    - This highlights a disconnect between visual text (e.g., packaging, auto-subtitles, text overlays) and emotionally resonant brand communication — especially in UGC or aesthetic-led videos.

    **Actionable Strategy:**

    - Provide creators with **predefined, branded subtitle templates or emotional keyword overlays** to enhance emotional relevance in on-screen text.
    - Avoid relying on OCR-only elements (e.g., packaging visuals or random on-screen phrases) to carry key campaign emotion — instead, pair them with strong narration or caption support.
    - Monitor the “Irrelevant” rate in OCR results as a **creative quality KPI**: a high rate may indicate visual message misalignment or missed brand storytelling opportunities.
""")


# Product Feedback Tab
with tab4:
    st.header("Product Feedback Insights")

    st.subheader("I. Product Tags Performance: Positive Sentiment Across Core Campaign Hashtags")
    
    
    st.markdown("""
    This chart analyzes **positive sentiment rates** across the **top-mentioned product tags** based on the `overall_sentiment` derived from all four models combined (Text, Audio, Video, OCR).  
    It helps identify which products receive the strongest positive emotional responses on TikTok.

    **Note:** We consider only product tags that appear frequently across the dataset to ensure reliability.
    """)

    # Efficient positive rate calculation for top product tags
    tag_counts = {}
    for _, row in merged_df_all[['combined_product_tags', 'overall_sentiment']].dropna().iterrows():
        tags = str(row['combined_product_tags']).lower().split('|')
        sentiment = row['overall_sentiment']
        for tag in tags:
            if tag not in tag_counts:
                tag_counts[tag] = {'total': 0, 'positive': 0}
            tag_counts[tag]['total'] += 1
            if sentiment == 'Positive':
                tag_counts[tag]['positive'] += 1

    # Convert to DataFrame
    tag_stats = pd.DataFrame([
        {'tag': tag, 'positive_rate': counts['positive'] / counts['total'], 'count': counts['total']}
        for tag, counts in tag_counts.items() if counts['total'] >= 10  # filter rare tags
    ])

    # Top 6 tags by frequency
    top_tag_stats = tag_stats.sort_values(by='count', ascending=False).head(6)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_tag_stats, x='tag', y='positive_rate', color='#8dd3c7', ax=ax)
    ax.set_title("Top Product Tags by Positive Sentiment Rate")
    ax.set_ylabel("Positive Sentiment Rate")
    ax.set_xlabel("Product Tag")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    st.markdown("""
    **Business Insight:**

    - All top product-related tags show **consistently moderate-to-high positive sentiment rates**, typically between 48% and 59%.
    - This consistency suggests that **Sol de Janeiro's core product content is emotionally resonant and well-received**, especially in fragrance-related mentions.
    - No product tag shows unusually low sentiment — indicating **no urgent negative perception risk** in popular SKUs.

    **Actionable Strategy:**

    - Continue to highlight fragrance-related tags such as `perfume` and `soldejaneiroperfume`, which slightly outperform others in positive sentiment.
    - Consider **standardizing campaign hashtags** to avoid fragmenting sentiment across similar tags (e.g., `soldejaneiro` vs. `soldejaneiroperfume`).
    - Track tag-level sentiment seasonally — even small variations may signal when certain SKUs (e.g., summer mists vs. winter creams) gain or lose traction.
    """)


    
    st.subheader("II. Monthly Positive Sentiment Trends by Product Tag")

    st.markdown("""
    This chart tracks how positive sentiment rates for **top product-related tags** evolve over time (by month).  
    It reveals potential **seasonal patterns**, **campaign effects**, and **tag-specific momentum**.

    Each data point represents the **proportion of content** tagged with a product that received a *positive* sentiment rating across the four models (text, audio, video, OCR).
    """)

    # Sample manageable data
    sample_df = merged_df_all[
        (merged_df_all['create_time_iso_y'].notna()) &
        (merged_df_all['combined_product_tags'].notna())
    ].sample(n=4000, random_state=42).copy()

    sample_df['create_time'] = pd.to_datetime(sample_df['create_time_iso_y'], errors='coerce')
    sample_df['month'] = sample_df['create_time'].dt.to_period('M').astype(str)
    sample_df['tag_list'] = sample_df['combined_product_tags'].str.lower().str.split('|')
    exploded = sample_df.explode('tag_list')

    top_tags = exploded['tag_list'].value_counts().head(5).index.tolist()
    filtered = exploded[exploded['tag_list'].isin(top_tags)]

    # Group by month + tag
    grouped = filtered.groupby(['month', 'tag_list'])['overall_sentiment'].value_counts().unstack().fillna(0)
    grouped['total'] = grouped.sum(axis=1)
    grouped['positive_rate'] = grouped.get('Positive', 0) / grouped['total']

    pivot = grouped.reset_index().pivot(index='month', columns='tag_list', values='positive_rate')

    # Plot
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    pivot.plot(marker='o', ax=ax2)
    ax2.set_title("Monthly Positive Sentiment Trends by Product Tag")
    ax2.set_ylabel("Positive Sentiment Rate")
    ax2.set_xlabel("Month")
    ax2.set_xticks(range(len(pivot.index)))
    ax2.set_xticklabels(pivot.index, rotation=45)
    ax2.set_ylim(0, 1)
    ax2.legend(title="Product Tag", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig2)

    st.markdown("""
    **Business Insight:**

    - Sentiment trends **fluctuate monthly by product**, revealing campaign cycles and seasonal preferences.
    - Tags like `perfume` or `soldejaneiroperfume` show **peaks in certain months**, aligning with likely promotional pushes.
    - Periods of lower sentiment may indicate content fatigue or mismatched messaging.

    **Actionable Strategy:**

    - Replicate high-sentiment month strategies: influencer mix, soundtracks, product positioning.
    - Analyze sentiment dips for **creative adjustments** or product repositioning.
    """)
