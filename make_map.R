setwd("~/Documents/GitHub/CS-Annotate/data/")

# load data
data = read.csv("deepchem_testset_predictions_update_2N1Q.txt", header = TRUE, sep = " ")
data$sasa = NULL 
colnames(data)[colnames(data)=="sasa.1"] = "sasa"
get_data_for_rna = function(data, rna = "5KH8"){
  # get data for a particular RNA
  return(data[data$id == rna, ])
}


filter_pucker = function(data){
  orig = data
  puckers = grepl("pucker", colnames(data))
  sub = data[, puckers]
  for (i in 1:nrow(data)){
    tmp = sub[i, ]
    row = rep(0, ncol(sub)) # ?? ncol(data)
    row[which.max(tmp)] = 1
    sub[i, ] = row
  }
  orig[, puckers] = sub
  return(orig)
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

generate_rna_maps = function(data, rna = "5KH8", cutoff = 0.50, width = 9, height = 9, pointsize = 10){
  # create maps for manuscript
  sub = get_data_for_rna(data, rna = rna)
  pred = as.matrix(filter_pucker(select_fingerprint(sub, predicted = TRUE))>cutoff)
  write.table(pred, paste0("pred_",rna,".csv"), col.names = T, row.names = F, quote = F)
  #pdf(file = sprintf("map_%s_predicted.pdf", rna), width = width, height = height, pointsize = pointsize)
  #print(lattice::levelplot(pred, las = 2, ylab = "Features", xlab = "Residues", useRaster = FALSE, pretty = TRUE, at = seq(0, 1, 0.005), col.regions = cols2 <- colorRampPalette(c("black", "blue", "green", "yellow",  "red"))(256)))
  #dev.off()
  
  actual = as.matrix(select_fingerprint(sub, predicted = FALSE)>cutoff)
  write.table(actual, paste0("actual_",rna,".csv"), col.names = T, row.names = F, quote = F)
  #pdf(file = sprintf("map_%s_actual.pdf", rna), width = width, height = height, pointsize = pointsize)
  #print(lattice::levelplot(actual, las = 2, ylab = "Features", xlab = "Residues", useRaster = FALSE, pretty = TRUE, at = seq(0, 1, 0.005), col.regions = cols2 <- colorRampPalette(c("black", "blue", "green", "yellow",  "red"))(256)))
  #dev.off()
  
  diff = abs(actual - pred)
  write.table(diff, paste0("diff_",rna,".csv"), col.names = T, row.names = F, quote = F)
  pdf(file = sprintf("map_%s_diff.pdf", rna), width = width, height = height, pointsize = pointsize)
  print(lattice::levelplot(diff, las = 2, ylab = "Features", xlab = "Residues", useRaster = FALSE, pretty = TRUE, at = seq(0, 1, 0.005), col.regions = cols2 <- colorRampPalette(c("black", "blue", "green", "yellow",  "red"))(256)))
  dev.off()
  
}


# generate maps for the fluoride riboswitch 5KH8
for(rna in c("5KH8","2JTP","2LU0","2N1Q")){
  generate_rna_maps(data, rna = rna)  
}

