# script to generate annotation maps
setwd("~/Downloads/")

# load data
data = read.csv("predictions.txt", header = TRUE, sep = " ")

get_data_for_rna = function(data, rna = "5KH8"){
  # get data for a particular RNA
  return(data[data$id == rna, ])
}

select_fingerprint = function(data, predicted = FALSE, names = c("sasa", "astack", "nastack", "pair", "syn_anti", "pucker_C2p_endo", "pucker_C3p_endo", "pucker_C2p_exo", "pucker_C3p_exo", "pucker_C1p_exo", "pucker_C4p_exo")){
  # generate a fingerprint
  if (predicted){
    tnames = paste("p", names, sep = "")
  } else {
    tnames = names
  }
  data = data[, tnames]
  colnames(data) = names
  return(data)
}

generate_rna_maps = function(data, rna = "5KH8", width = 9, height = 9, pointsize = 10){
  # create maps for manuscript
  sub = get_data_for_rna(data, rna = rna)
  pred = as.matrix(select_fingerprint(sub, predicted = TRUE)>0.5)
  pdf(file = sprintf("map_%s_predicted.pdf", rna), width = width, height = height, pointsize = pointsize)
  print(lattice::levelplot(pred, las = 2, ylab = "Features", xlab = "Residues", useRaster = FALSE, pretty = TRUE, at = seq(0, 1, 0.005), col.regions = cols2 <- colorRampPalette(c("black", "blue", "green", "yellow",  "red"))(256)))
  dev.off()
  
  actual = as.matrix(select_fingerprint(sub, predicted = FALSE)>0.5)
  pdf(file = sprintf("map_%s_actual.pdf", rna), width = width, height = height, pointsize = pointsize)
  print(lattice::levelplot(actual, las = 2, ylab = "Features", xlab = "Residues", useRaster = FALSE, pretty = TRUE, at = seq(0, 1, 0.005), col.regions = cols2 <- colorRampPalette(c("black", "blue", "green", "yellow",  "red"))(256)))
  dev.off()

  diff = abs(actual - pred)
  pdf(file = sprintf("map_%s_diff.pdf", rna), width = width, height = height, pointsize = pointsize)
  print(lattice::levelplot(diff, las = 2, ylab = "Features", xlab = "Residues", useRaster = FALSE, pretty = TRUE, at = seq(0, 1, 0.005), col.regions = cols2 <- colorRampPalette(c("black", "blue", "green", "yellow",  "red"))(256)))
  dev.off()
  
}

# generate maps for the fluoride riboswitch 5KH8
generate_rna_maps(data, rna = "5KH8")

