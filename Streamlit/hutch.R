## Package admin

library(dplyr)
library(lubridate)
library(ggplot2)
library(scales)

## Read in the Part 1 datasets

activity <- read.csv("C:/Users/44798/Desktop/Hutch Task/Game Analyst Task/Part 1, ACTIVITY.csv")
cohort   <- read.csv("C:/Users/44798/Desktop/Hutch Task/Game Analyst Task/Part 1, COHORT.csv")
part2    <- read.csv("C:/Users/44798/Desktop/Hutch Task/Game Analyst Task/Part 2, DATA.csv")

## Merge/Clean the Part 1 data, and filter out users who installed outside of the test range 14/12/2022 to  11/01/2023

part1 <- merge(activity, cohort, by="USER_ID")
part1$ACTIVITY_DATE <- as.Date(part1$ACTIVITY_DATE, format = "%d/%m/%Y")
part1$INSTALL_DATE <- as.Date(part1$INSTALL_DATE, format = "%d/%m/%Y")
part2$Date <- as.Date(part2$Date, format = "%d/%m/%Y")

part1 <- part1 %>%
  filter(INSTALL_DATE >= as.Date("2022-12-14") &
           INSTALL_DATE <= as.Date("2023-01-11"))

part1 <- part1 %>%
  mutate(
    day_after_install = as.numeric(part1$ACTIVITY_DATE - part1$INSTALL_DATE)
  )

part1 <- part1 %>%
   filter(day_after_install >= 0) 

part2 <- part2 %>%
  mutate(
    UA.Spend = as.numeric(gsub("[$,]", "", UA.Spend))
  )

part2 <- part2 %>%
  mutate(
    Total.Revenue = as.numeric(gsub("[$,]", "",  Total.Revenue))
  )

part2 <- part2 %>%
  mutate(
    Installs = as.numeric(gsub(",", "",  Installs))
  )%>%
  filter(UA.Spend > 0)



## Add D1
user_installs <- part1 %>%
  group_by(USER_ID, COHORT_NAME) %>%
  summarise(INSTALL_DATE = min(INSTALL_DATE), .groups = "drop")

d1_flag <- part1 %>%
  inner_join(user_installs, by = c("USER_ID", "COHORT_NAME")) %>%
  mutate(days_after_install = as.numeric(part1$ACTIVITY_DATE - part1$INSTALL_DATE)) %>%
  group_by(USER_ID, COHORT_NAME) %>%
  summarise(d1_retained = any(days_after_install == 1), .groups = "drop")


d1_retention <- d1_flag %>%
  group_by(COHORT_NAME) %>%
  summarise(
    users = n(),
    retained_users = sum(d1_retained),
    d1_retention_rate = retained_users / users
  )

## Add D7
d7_flag <- part1 %>%
  inner_join(user_installs, by = c("USER_ID", "COHORT_NAME")) %>%
  mutate(days_after_install = as.numeric(part1$ACTIVITY_DATE - part1$INSTALL_DATE)) %>%
  group_by(USER_ID, COHORT_NAME) %>%
  summarise(d1_retained = any(days_after_install == 7), .groups = "drop")


d7_retention <- d7_flag %>%
  group_by(COHORT_NAME) %>%
  summarise(
    users = n(),
    retained_users = sum(d1_retained),
    d7_retention_rate = retained_users / users
  )

## Add D30

d30_flag <- part1 %>%
  inner_join(user_installs, by = c("USER_ID", "COHORT_NAME")) %>%
  mutate(days_after_install = as.numeric(part1$ACTIVITY_DATE - part1$INSTALL_DATE)) %>%
  group_by(USER_ID, COHORT_NAME) %>%
  summarise(d1_retained = any(days_after_install == 30), .groups = "drop")


d30_retention <- d30_flag %>%
  group_by(COHORT_NAME) %>%
  summarise(
    users = n(),
    retained_users = sum(d1_retained),
    d30_retention_rate = retained_users / users
  )


## Do a prop test for D1

d1_test <- prop.test(d1_retention$retained_users, d1_retention$users)
d7_test <- prop.test(d7_retention$retained_users, d7_retention$users)
d30_test <- prop.test(d30_retention$retained_users, d30_retention$users)


## LTV curves with SE bars
# Per-user cumulative spend by day
user_ltv <- part1 %>%
  group_by(COHORT_NAME, USER_ID, day_after_install) %>%
  summarise(daily_spend = sum(DAILY_REVENUE_USD, na.rm=TRUE), .groups="drop") %>%
  arrange(COHORT_NAME, USER_ID, day_after_install) %>%
  group_by(COHORT_NAME, USER_ID,) %>%
  mutate(cumulative_spend = cumsum(daily_spend)) %>%
  ungroup()

ltv_summary <- user_ltv %>%
  group_by(COHORT_NAME, day_after_install) %>%
  summarise(
    mean_ltv = mean(cumulative_spend),
    se_ltv = sd(cumulative_spend) / sqrt(n()),
    .groups = "drop"
  ) %>%
  mutate(
    ltv_lower = mean_ltv - 1.96*se_ltv, 
    ltv_upper = mean_ltv + 1.96*se_ltv
  ) %>%
  filter(day_after_install >= 0) 

ltv_30 <- ltv_summary %>%
  filter(day_after_install <= 30)

ggplot(ltv_30, aes(x = day_after_install, y = mean_ltv, color = COHORT_NAME)) +
  geom_line(size = 1.2) +
  geom_ribbon(aes(ymin = ltv_lower, ymax = ltv_upper, fill = COHORT_NAME),
              alpha = 0.2, color = NA) +
  labs(
    x = "Days After Install",
    y = "Cumulative LTV per User (USD)",
    title = "Cohort LTV (First 30 Days) with 95% Confidence Intervals"
  ) +
  theme(
    axis.title = element_text(size = 14),   
    axis.text = element_text(size = 12),
    plot.title   = element_text(size = 16, face = "bold", hjust = 0.5)
  ) +
  scale_y_continuous(labels = label_dollar()) +
  scale_color_discrete(
    name = "Test Group",
    labels = c("Control", "Variant")
  ) +
  scale_fill_discrete(
    name = "Test Group",
    labels = c("Control", "Variant")
  ) +
  xlim(0, 30)


## Let's look at engagement now

# Per-user cumulative engagement per day
user_lte <- part1 %>%
  group_by(COHORT_NAME, USER_ID, day_after_install) %>%
  summarise(daily_play = sum(DAILY_MATCHES_PLAYED, na.rm=TRUE), .groups="drop") %>%
  arrange(COHORT_NAME, USER_ID, day_after_install) %>%
  group_by(COHORT_NAME, USER_ID,) %>%
  mutate(cumulative_play = cumsum(daily_play)) %>%
  ungroup()

lte_summary <- user_lte %>%
  group_by(COHORT_NAME, day_after_install) %>%
  summarise(
    mean_lte = mean(cumulative_play),
    se_lte = sd(cumulative_play) / sqrt(n()),
    .groups = "drop"
  ) %>%
  mutate(
    lte_lower = mean_lte - 1.96*se_lte,  
    lte_upper = mean_lte + 1.96*se_lte
  ) %>%
  filter(day_after_install >= 0) 

lte_30 <- lte_summary %>%
  filter(day_after_install <= 30)

ggplot(lte_30, aes(x = day_after_install, y = mean_lte, color = COHORT_NAME)) +
  geom_line(size = 1.2) +
  geom_ribbon(aes(ymin = lte_lower, ymax = lte_upper, fill = COHORT_NAME),
              alpha = 0.2, color = NA) +
  labs(
    x = "Days After Install",
    y = "Cumulative Matches per User",
    title = "Cohort MPU (First 30 Days) with 95% Confidence Intervals"
  ) +
  theme(
    axis.title = element_text(size = 14),   
    axis.text = element_text(size = 12),
    plot.title   = element_text(size = 16, face = "bold", hjust = 0.5)
  ) +
  scale_color_discrete(
    name = "Test Group",
    labels = c("Control", "Variant")
  ) +
  scale_fill_discrete(
    name = "Test Group",
    labels = c("Control", "Variant")
  ) +
  xlim(0, 30)



## CPI / ROAS

# CPI/ROAS calculations
cpi <- part2 %>%
  mutate(
    
    daily_cpi = part2$UA.Spend / part2$Installs,
    
    # Cumulative sums
    cumulative_ua_spend = cumsum(UA.Spend),
    cumulative_installs = cumsum(Installs),
    cumulative_revenue =  cumsum(Total.Revenue),
    
    # Cumulative CPI: NA if cumulative installs = 0
    cumulative_cpi = ifelse(cumulative_installs == 0, NA,
                            cumulative_ua_spend / cumulative_installs),
    
    # cumulative ltv
    cumulative_ltv = cumulative_revenue/ cumulative_installs,
    
    #cumulative roas
    cumulative_roas = cumulative_ltv / cumulative_cpi
  )

#c-ROAS
ggplot(cpi, aes(x = Date, y = cumulative_roas, color = "red")) +
  geom_line(size = 1.2) +
  theme(legend.position = "none") +
  geom_hline(yintercept = 1.2, color = "red", linetype = "dashed", size = 1) +
  annotate(
    "text",
    x = as.Date("2022-04-02"),               
    y = 1.2,             
    label = "120%",      
    color = "red",
    size = 4,           
    vjust = -0.5         
  ) +
  scale_y_continuous(labels = label_percent()) +
  labs(
    x = "Date",
    y = "Cumulative ROAS",
    title = "New Game - Cumulative Return on Ad Spend"
  ) +
  theme(
    axis.title = element_text(size = 14),   
    axis.text = element_text(size = 12),
    plot.title   = element_text(size = 16, face = "bold", hjust = 0.5)
  ) 

#c-CPI
ggplot(cpi, aes(x = Date, y = cumulative_cpi, color = "red")) +
  geom_line(size = 1.2) +
  theme(legend.position = "none") +
  scale_y_continuous(labels = label_dollar()) + 
  labs(
    x = "Date",
    y = "Cumulative CPI",
    title = "Cumulative CPI (USD)"
  ) +
  theme(
    axis.title = element_text(size = 14),   
    axis.text = element_text(size = 12),
    plot.title   = element_text(size = 16, face = "bold", hjust = 0.5)
  ) 

#c-LTV
ggplot(cpi, aes(x = Date, y = cumulative_ltv, color = "red")) +
  geom_line(size = 1.2) +
  theme(legend.position = "none") +
  scale_y_continuous(labels = label_dollar()) + 
  labs(
    x = "Date",
    y = "Cumulative LTV",
    title = "New Game - Cumulative LTV (USD)"
  ) +
  theme(
    axis.title = element_text(size = 14),   
    axis.text = element_text(size = 12),
    plot.title   = element_text(size = 16, face = "bold", hjust = 0.5)
  ) 