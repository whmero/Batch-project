library(ggplot2)


df <- read.csv("partsComparison.csv")
df$Model <- as.factor(df$Model)


# List of numeric columns to visualize
numeric_columns <- c("frames_to_process", "flops", "total_processing_time",
                     "single_frame_time", "left_output_size",
                     "avg_consec_inference_gap", "total_left_model_time",
                     "total_right_model_time")

# Create a list to store box plots
box_plots <- list()

# Create box plots for each numeric variable
for (col in numeric_columns) {
  plot <- ggplot(data = df, aes(x = Model, y = .data[[col]], fill = Model)) +
    geom_boxplot() +
    labs(title = paste("Box Plot for", col, "by Model"),
         x = "Model",
         y = col)
  
  box_plots[[col]] <- plot
}

# Print the box plots
for (i in 1:length(box_plots)) {
  print(box_plots[[i]])
}


# List of numeric columns to visualize
numeric_columns <- c("frames_to_process", "flops", "total_processing_time",
                     "single_frame_time", "left_output_size",
                     "avg_consec_inference_gap", "total_left_model_time",
                     "total_right_model_time")

# Create a list to store scatter plots
scatter_plots <- list()

# Create scatter plots for each numeric variable
for (col in numeric_columns) {
  plot <- ggplot(data = df, aes(x = split_no, y = .data[[col]], color = Model)) +
    geom_point() +
    labs(title = paste("Scatter Plot of", col, "vs. split_no by Model"),
         x = "split_no",
         y = col)
  
  scatter_plots[[col]] <- plot
}

# Print the scatter plots
for (i in 1:length(scatter_plots)) {
  print(scatter_plots[[i]])
}

# Function to create a bar chart for a numeric column
create_bar_chart <- function(data, col_name) {
  ggplot(data = data, aes(x = Model, y = .data[[col_name]], fill = Model)) +
    geom_bar(stat = "identity") +
    labs(title = paste("Bar Chart for", col_name, "by Model"),
         x = "Model",
         y = col_name) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Create and display bar charts for all numeric columns
for (col in numeric_columns) {
  bar_chart <- create_bar_chart(df, col)
  print(bar_chart)
}


# Function to create a histogram for a numeric column with different colors for Models
create_histogram <- function(data, col_name) {
  ggplot(data = data, aes(x = .data[[col_name]], fill = Model)) +
    geom_histogram(binwidth = 0.5, position = "identity", alpha = 0.5) +
    labs(title = paste("Histogram for", col_name),
         x = col_name,
         y = "Frequency")
}

# Create and display histograms for all numeric columns
for (col in numeric_columns) {
  histogram <- create_histogram(df, col)
  print(histogram)
}


# Function to create a violin plot for a numeric column
create_violin_plot <- function(data, col_name) {
  ggplot(data = data, aes(x = Model, y = .data[[col_name]], fill = Model)) +
    geom_violin(trim = FALSE) +
    labs(title = paste("Violin Plot for", col_name, "by Model"),
         x = "Model",
         y = col_name)
}

# Create and display violin plots for all numeric columns
for (col in numeric_columns) {
  violin_plot <- create_violin_plot(df, col)
  print(violin_plot)
}