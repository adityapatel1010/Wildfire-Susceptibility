var states=ee.FeatureCollection("FAO/GAUL/2015/level1");
var utk=states.filter(ee.Filter.eq('ADM1_NAME','Uttarakhand'));

var b_d=ee.Date('2018-1-1');
var e_d=ee.Date('2018-6-30');
var lst_fun=function(s_d,f_d){
  var lst=ee.ImageCollection("MODIS/061/MOD11A1").select('LST_Day_1km').filterDate(s_d, f_d).filterBounds(utk);
  lst = lst.reduce(ee.Reducer.mean()).multiply(0.02).add(-273.15).rename('LST'); //56.85  -13.15
  return lst;
  }
var fire_fun=function(s_d,f_d){
  var fire = ee.ImageCollection("MODIS/061/MCD64A1").filterDate(s_d, f_d).filterBounds(utk).select('BurnDate').max().rename('fire');
  fire=fire.divide(fire);
  return fire;
  }

var air_fun=function(s_d,f_d){
  var air_temp = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterDate(s_d,f_d).select('temperature_2m').filterBounds(utk).max().rename('air_temp');
  return air_temp;
}

var ndvi_fun=function(s_d,f_d){
  var ndvi = ee.ImageCollection('MODIS/061/MOD13A1').filter(ee.Filter.date(s_d,f_d)).select('NDVI').max().rename('ndvi');
  return ndvi;
}


var rain_fun=function(s_d,f_d){
  var precipitation = ee.ImageCollection('JAXA/GPM_L3/GSMaP/v6/operational').filter(ee.Filter.date(s_d,f_d)).select('hourlyPrecipRate').max().rename('rain');
  return precipitation;
}

//-----------------------------------

var lst=lst_fun(b_d,e_d);
var fire=fire_fun(b_d,e_d);
var ndvi=ndvi_fun(b_d,e_d);
var precipitation=rain_fun(b_d,e_d);
var air_temp=air_fun(b_d,e_d);
var pop = ee.Image("JRC/GHSL/P2023A/GHS_POP/2030").rename('pop');
var temp=ee.Image("JRC/GHSL/P2023A/GHS_POP/2030");
temp=temp.subtract(temp).rename('fire');
var landforms = ee.Image('CSP/ERGo/1_0/Global/ALOS_landforms').select('constant').rename('landforms');

var combined = ee.Image([lst,air_temp,ndvi,landforms,pop,fire]);

var sample = combined.sample({
  numPixels: 10000,
  scale : 100,
  region : utk,
  geometries: true
}).randomColumn();
Map.addLayer(sample, {}, 'FireSample', true);

var train = sample.filter(ee.Filter.lte('random', 0.8));
var test = sample.filter(ee.Filter.gt('random', 0.8));

combined = ee.Image([lst,air_temp,ndvi,landforms,pop,temp]);

sample = combined.sample({
  numPixels: 100,
  scale : 100,
  region : utk,
  geometries: true
}).randomColumn();
Map.addLayer(sample, {color:'blue'}, 'TempSample', true);

train=train.merge(sample.filter(ee.Filter.lte('random', 0.8)));
test=test.merge(sample.filter(ee.Filter.gt('random', 0.8)));

//=====================

b_d=b_d.advance(6, 'months');
e_d=e_d.advance(6, 'months');
lst=lst_fun(b_d,e_d);
fire=fire_fun(b_d,e_d);
ndvi=ndvi_fun(b_d,e_d);
precipitation=rain_fun(b_d,e_d);
air_temp=air_fun(b_d,e_d);

combined = ee.Image([lst,air_temp,ndvi,landforms,pop,fire]);

sample = combined.sample({
  numPixels: 10000,
  region : utk,
  scale: 100,
  geometries: true
}).randomColumn();

train=train.merge(sample.filter(ee.Filter.lte('random', 0.8)));
test=test.merge(sample.filter(ee.Filter.gt('random', 0.8)));

combined = ee.Image([lst,air_temp,ndvi,landforms,pop,temp]);

sample = combined.sample({
  numPixels: 100,
  region : utk,
  scale: 100,
  geometries: true
}).randomColumn();

train=train.merge(sample.filter(ee.Filter.lte('random', 0.8)));
test=test.merge(sample.filter(ee.Filter.gt('random', 0.8)));

//--------------------------------

// print("Training size : ",train.size());
// print("Testing size : ",test.size());

// Export.table.toDrive({
//   collection: test.filter('fire > 0 '),
//   description : 'TEST1_FIRE',
//   fileFormat: 'SHP'
// });

// Export.table.toDrive({
//   collection: test.filter('fire == 0 '),
//   description : 'TEST0_FIRE',
//   fileFormat: 'SHP'
// });

train=train.select(['LST','air_temp','landforms','ndvi','pop','fire']);
test=test.select(['LST','air_temp','landforms','ndvi','pop']);
print("Training Data : ",train);
print("Testing Data : ",test);

// Export.table.toDrive({
//   collection: train,
//   description : 'Train_UTK',
//   fileFormat: 'SHP'
// });




var rf = ee.Classifier.smileRandomForest(10).train(train, 'fire', ['LST','air_temp','landforms','ndvi','pop']).setOutputMode('CLASSIFICATION');
// print('Regression RF', regression.explain());
var result_rf = test.classify(rf, 'Fire_Prediction_RF');
var acc_rf = rf.confusionMatrix();
print('RF Resubstitution error matrix: ', acc_rf);
print('RF Overall accuracy: ', acc_rf.accuracy());

Export.table.toDrive({
  collection: result_rf.filter('Fire_Prediction_RF > 0'),
  description : 'PRED1_UTK',
  fileFormat: 'SHP'
});

Export.table.toDrive({
  collection: result_rf.filter('Fire_Prediction_RF == 0'),
  description : 'PRED0_UTK',
  fileFormat: 'SHP'
});

var svm = ee.Classifier.libsvm({kernelType: 'RBF'}).train(train, 'fire', ['LST','air_temp','landforms','ndvi','pop']);
var result_svm = test.classify(svm);
var acc_svm = svm.confusionMatrix();
print('SVM Resubstitution error matrix: ', acc_svm);
print('SVM Overall accuracy: ', acc_svm.accuracy());

var cart = ee.Classifier.smileCart().train(train, 'fire', ['LST','air_temp','landforms','ndvi','pop']);
var result_cart = test.classify(cart);
var acc_cart = cart.confusionMatrix();
print('CART Resubstitution error matrix: ', acc_cart);
print('CART Overall accuracy: ', acc_cart.accuracy());

var naive = ee.Classifier.smileNaiveBayes().train(train, 'fire', ['LST','air_temp','landforms','ndvi','pop']);
var result_naive = test.classify(naive);
var acc_naive = naive.confusionMatrix();
print('NB Resubstitution error matrix: ', acc_naive);
print('NB Overall accuracy: ', acc_naive.accuracy());

var gtb = ee.Classifier.smileGradientTreeBoost(10).train(train, 'fire', ['LST','air_temp','landforms','ndvi','pop']);
var result_gtb = test.classify(gtb);
var acc_gtb = gtb.confusionMatrix();
print('GTB Resubstitution error matrix: ', acc_gtb);
print('GTB Overall accuracy: ', acc_gtb.accuracy());

var importance = ee.Dictionary(rf.explain().get('importance'));
var totalImportance = importance.values().reduce(ee.Reducer.sum());
var importancePercentage = importance.map(function (band,importance) {
  return ee.Number(importance).divide(totalImportance).multiply(100);
})
print(importancePercentage)
var importance = ee.Dictionary(gtb.explain().get('importance'));
var totalImportance = importance.values().reduce(ee.Reducer.sum());
var importancePercentage = importance.map(function (band,importance) {
  return ee.Number(importance).divide(totalImportance).multiply(100);
})
print(importancePercentage)
var importance = ee.Dictionary(cart.explain().get('importance'));
var totalImportance = importance.values().reduce(ee.Reducer.sum());
var importancePercentage = importance.map(function (band,importance) {
  return ee.Number(importance).divide(totalImportance).multiply(100);
})
print(importancePercentage)
