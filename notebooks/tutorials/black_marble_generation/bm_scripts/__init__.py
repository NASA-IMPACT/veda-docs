try:
    import gdal
except ImportError:
    from osgeo import gdal
    
import glob
import numpy as np
import os
from osgeo import ogr
import shapely.geometry
import shapely.wkt
import boto3
from datetime import datetime
from datetime import date
import utm
from numpy import inf
import matplotlib.pyplot as plt
import rasterio
import rasterio.features
import fiona
import fiona.transform
import requests
import osmnx as ox
import os.path
from pyproj import Proj
import json
import utm
from geopy.distance import geodesic
from rasterio.io import MemoryFile
from rio_cogeo.profiles import cog_profiles
from rio_cogeo.cogeo import cog_translate
import pprint
import math

class BMDataManager:

    def copy_file(self, source_bucket, source_path, destination_bucket, destination_path):
        session = self.assumed_role_session()
        s3 = session.resource('s3')
        copy_source = {
            'Bucket': source_bucket,
            'Key': source_path
         }
        s3.meta.client.copy(copy_source, destination_bucket, destination_path)

    def get_upload_file_name(self, lat1, lat2, long1, long2, year, month, day, prefix = ""):
        nw = (math.ceil(max(lat1, lat2)) , math.floor(min(long1,long2)))
        se = (math.floor(min(lat1, lat2)), math.ceil(max(long1, long2)))

        file_name = 'hdnightlights_' + str(year) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2) + '_'
        file_name = file_name +  str(nw[0]) + 'N' if nw[0] > 0 else str(-nw[0]) + 'S'
        file_name = file_name +  str(nw[1]) + 'E' if nw[1] > 0 else str(-nw[1]) + 'W'

        file_name = file_name +  str(se[0]) + 'N' if se[0] > 0 else str(-se[0]) + 'S'
        file_name = file_name +  str(se[1]) + 'E' if se[1] > 0 else str(-se[1]) + 'W'
        file_name = file_name + '-day.tif'

        return prefix + file_name

    def download_file(self, source_bucket, source_path, destination_path):
        session = self.assumed_role_session()
        s3 = session.client('s3')
        with open(destination_path, 'wb') as f:
            s3.download_file(source_bucket, source_path,
            destination_path, ExtraArgs = {"RequestPayer": "requester"})

    def assumed_role_session(self):
        role = boto3.client('sts').assume_role_with_web_identity(RoleArn=self.aws_role_arn, RoleSessionName='assume-role', WebIdentityToken=self.aws_id_token)
        credentials = role['Credentials']
        aws_access_key_id = credentials['AccessKeyId']
        aws_secret_access_key = credentials['SecretAccessKey']
        aws_session_token = credentials['SessionToken']
        return boto3.session.Session(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, aws_session_token=aws_session_token)
    
    
class GISUtils:
    def get_geo(self, f, band):
        ds = gdal.Open(f, gdal.GA_ReadOnly)
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        img = ds.GetRasterBand(band).ReadAsArray(0,0,cols,rows)
        in_geo = ds.GetGeoTransform()
        projref = ds.GetProjectionRef()
        ds = None
        return img, in_geo, projref

    def save_image(self, f,in_geo,projref,type,out):
        #create New Raster
        driver = gdal.GetDriverByName('GTiff')
        if driver == None:
            print ("Failed to find the gdal driver")
            exit()
        newRaster = driver.Create(out, f.shape[1], f.shape[0], 1, type)
        newRaster.SetProjection(projref)
        newRaster.SetGeoTransform(in_geo)
        outBand = newRaster.GetRasterBand(1)
        outBand.SetNoDataValue(0)
        outBand.WriteArray(f, 0, 0)
        driver = None
        outBand = None
        newRaster = None

    def scale_image(self, inputarry, min_value, max_value):
        inputarry = np.where(inputarry > max_value, max_value,inputarry)
        inputarry = np.where(inputarry < min_value, min_value,inputarry)
        new_arr = ((inputarry - inputarry.min()) * (1/(inputarry.max() - inputarry.min()) * 255)).astype('uint8')
        return new_arr

    def save_geotiff_rgb(self, filename, r, g, b, projref, in_geo):
      # Write a GeoTIFF
      format = 'GTiff'
      rows, cols = np.shape(r)
      driver = gdal.GetDriverByName(format)
      out_ds = driver.Create(filename, cols, rows, 3, gdal.GDT_Byte)
      out_ds.SetProjection(projref)
      out_ds.SetGeoTransform(in_geo)
      out_ds.GetRasterBand(1).WriteArray(r)
      out_ds.GetRasterBand(2).WriteArray(g)
      out_ds.GetRasterBand(3).WriteArray(b)
      out_ds = None

    def save_geotiff_single(self, filename, band_data, projref, in_geo):
      # Write a GeoTIFF
      format = 'GTiff'
      rows, cols = np.shape(band_data)
      driver = gdal.GetDriverByName(format)
      out_ds = driver.Create(filename, cols, rows, 1, gdal.GDT_UInt16)
      out_ds.SetProjection(projref)
      out_ds.SetGeoTransform(in_geo)
      out_ds.GetRasterBand(1).WriteArray(band_data)
      out_ds = None

    def subset_raster_normal(self, inputRaster, outputRaster, minX, maxY, maxX, minY, scale):
      ds = gdal.Open(inputRaster)
      #extended BBox
      top1 = maxY
      left1 = minX
      bottom1 = minY
      right1 = maxX
      ds = gdal.Translate('new1.tif', inputRaster, projWin = [left1, top1, right1, bottom1], resampleAlg='bilinear')
      ds = None
      ds = gdal.Open('new1.tif')
      ds = gdal.Warp('new2.tif', ds, xRes=scale, yRes=scale, resampleAlg='bilinear')
      ds = None
      ds = gdal.Open('new2.tif')
      ds = gdal.Translate(outputRaster, ds, projWin = [minX, maxY, maxX, minY], resampleAlg='bilinear')
      ds = None
      os.remove('new1.tif')
      os.remove('new2.tif')
        
    def wkt_to_json(self, wkt_string):
        # Create a Projection object from the WKT string
        proj = Proj(wkt_string)
        # Convert the WKT string to a dictionary representation
        json_dict = proj.crs.to_json_dict()
        return json_dict

    def convert_to_wgs84(self, input, output, long1, lat1, long2, lat2):
        ds = gdal.Open(input)
        srs_wkt = ds.GetProjection()
        json_data = self.wkt_to_json(srs_wkt)
        src_srs = json_data['id']['code']
        #ds = gdal.Warp(output, ds, 
        #               resampleAlg='bilinear', warpMemoryLimit = 4000, multithread=True, srcSRS = "EPSG:32617", dstSRS = "EPSG:4326")
        ds = gdal.Warp("outputs/conv.tiff", ds, 
                       resampleAlg='bilinear', 
                       warpMemoryLimit = 4000, multithread=True,
                       # outputBounds=[min(long1, long2), min(lat1, lat2), max(long1, long2), max(lat1, lat2)],
                       srcSRS = "EPSG:" + str(src_srs),
                       dstSRS = "EPSG:4326")
        ds = None
        min_x = min(long1, long2)
        max_x = max(long1, long2)
        min_y = min(lat1, lat2)
        max_y = max(lat1, lat2)
        ds = gdal.Open("outputs/conv.tiff")
        ds = gdal.Translate(output, ds, projWin = [min_x, max_y, max_x, min_y], resampleAlg='bilinear')
        ds = None

    def crop_to_bb_utm(self, input, output, x1, y1, x2, y2):
        ds = gdal.Open(input)
        ## Keep some margin for cropping
        min_x = min(x1, x2)  - 1000
        max_x = max(x1, x2) + 1000
        min_y = min(y1, y2) - 1000
        max_y = max(y1, y2) + 1000
        ds = gdal.Translate(output, ds, projWin = [min_x, max_y, max_x, min_y], resampleAlg='bilinear')
        ds = None

    def get_utm_zone(self, latitude, longitude):
        utm_info = utm.from_latlon(latitude, longitude)
        return utm_info
    
    def calculate_bounding_box_area(self, lat1, long1, lat2, long2):
        
        min_lon = min(long1, long2)
        min_lat = min(lat1, lat2)
        max_lon = max(long1, long2)
        max_lat = max(lat1, lat2)
        
        sw_point = (min_lat, min_lon)
        ne_point = (max_lat, max_lon)
        width = geodesic(sw_point, (max_lat, min_lon)).meters
        height = geodesic(sw_point, (min_lat, max_lon)).meters
        area = width * height
        return area / (1000 * 1000)
    
    def fill_gaps(self, pxs):
        max_shape = np.max([matrix.shape for matrix in pxs], axis=0)
        unified_matrices = np.zeros((len(pxs), *max_shape), dtype=np.float64)
        for idx, matrix in enumerate(pxs):
            unified_matrices[idx, :matrix.shape[0], :matrix.shape[1]] = matrix
        return unified_matrices
    
    def crop_products_to_bb(self, lat1, long1, lat2, long2, b3_outputs, b4_outputs, b5_outputs, b3_p, b3_r, b3_a):
        x1, y1, _, _ = self.get_utm_zone(lat1,long1)
        x2, y2, _, _ = self.get_utm_zone(lat2,long2)

        b3_final = []
        b4_final = []
        b5_final = []
        p_final = []
        r_final = []
        a_final = []

        for i in range(len(b3_outputs)):
            self.crop_to_bb_utm(b3_outputs[i], 'outputs/B3_' + str(i) + '.TIFF', x1, y1, x2, y2)
            b3_final.append('outputs/B3_' + str(i) + '.TIFF')

        for i in range(len(b4_outputs)):
            self.crop_to_bb_utm(b4_outputs[i], 'outputs/B4_' + str(i) + '.TIFF', x1, y1, x2, y2)
            b4_final.append('outputs/B4_' + str(i) + '.TIFF')

        for i in range(len(b5_outputs)):
            self.crop_to_bb_utm(b5_outputs[i], 'outputs/B5_' + str(i) + '.TIFF', x1, y1, x2, y2)
            b5_final.append('outputs/B5_' + str(i) + '.TIFF')

        for i in range(len(b3_p)):
            self.crop_to_bb_utm(b3_p[i], 'outputs/P_' + str(i) + '.TIFF', x1, y1, x2, y2)
            p_final.append('outputs/P_' + str(i) + '.TIFF')

        for i in range(len(b3_r)):
            self.crop_to_bb_utm(b3_r[i], 'outputs/R_' + str(i) + '.TIFF', x1, y1, x2, y2)
            r_final.append('outputs/R_' + str(i) + '.TIFF')

        for i in range(len(b3_a)):
            self.crop_to_bb_utm(b3_a[i], 'outputs/A_' + str(i) + '.TIFF', x1, y1, x2, y2)
            a_final.append('outputs/A_' + str(i) + '.TIFF')

        return b3_final, b4_final, b5_final, p_final, r_final, a_final
   
# https://github.com/robintw/LatLongToWRS/blob/master/get_wrs.py

class LandsatUtils:

    def __init__(self, aws_access_key, aws_secret_key, aws_session_token, shapefile="./WRS2_descending.shp"):
        
        
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.aws_session_token = aws_session_token
        # Open the shapefile
        self.shapefile = ogr.Open(shapefile)
        # Get the only layer within it
        self.layer = self.shapefile.GetLayer(0)
        self.polygons = []
        # For each feature in the layer
        for i in range(self.layer.GetFeatureCount()):
            # Get the feature, and its path and row attributes
            feature = self.layer.GetFeature(i)
            path = feature['PATH']
            row = feature['ROW']
            # Get the geometry into a Shapely-compatible
            # format by converting to Well-known Text (Wkt)
            # and importing that into shapely
            geom = feature.GetGeometryRef()
            shape = shapely.wkt.loads(geom.ExportToWkt())
            # Store the shape and the path/row values
            # in a list so we can search it easily later
            self.polygons.append((shape, path, row))


    def get_wrs(self, lat, lon):
        
        # Create a point with the given latitude
        # and longitude (NB: the arguments are lon, lat
        # not lat, lon)
        pt = shapely.geometry.Point(lon, lat)
        res = []
        # Iterate through every polgon
        for poly in self.polygons:
            # If the point is within the polygon then
            # append the current path/row to the results
            # list
            if pt.within(poly[0]):
                res.append({'path': poly[1], 'row': poly[2]})
        # Return the results list to the user
        return res

    def filter_landsat_data_dir(self, year, month, day, path, row):

        year = str(year)
        month = str(month).zfill(2)
        day = str(day).zfill(2)
        session = boto3.Session(aws_access_key_id= self.aws_access_key, 
                                aws_secret_access_key= self.aws_secret_key, 
                                aws_session_token=self.aws_session_token)
        s3 = session.client('s3')
        directories = s3.list_objects_v2(
            Bucket='usgs-landsat',
            Prefix='collection02/level-2/standard/oli-tirs/' + str(year) + '/' + path + '/' + row + '/',
            RequestPayer='requester',
            Delimiter='/'
        )['CommonPrefixes']
        selected_start_date = None
        selected_end_date = None
        selected_prefix = None
        for entry in directories:
            date_range = entry['Prefix'].split('/')[-2].split('_')[3:5]
            target_date = datetime.strptime(year + month + day, '%Y%m%d')
            start_date = datetime.strptime(date_range[0], '%Y%m%d')
            end_date = datetime.strptime(date_range[1], '%Y%m%d')
            if target_date >= start_date and target_date <= end_date:
                if selected_start_date:
                    if selected_start_date < start_date and selected_end_date > end_date:
                        selected_prefix = entry['Prefix']
                        selected_start_date = start_date
                        selected_end_date = end_date
                else:    
                    selected_prefix = entry['Prefix']
                    selected_start_date = start_date
                    selected_end_date = end_date
        return selected_prefix
    
    def download_landsat_band(self, dir, band, output_path, ignore_qf):
        if str(band) == 'QA_PIXEL':
            if ignore_qf:
                return
            band_file = dir + dir.split('/')[-2] + "_QA_PIXEL.TIF"
            print("Downloading QA_PIXEL ", band_file)
        elif str(band) == 'QA_RADSAT':
            band_file = dir + dir.split('/')[-2] + "_QA_RADSAT.TIF"
            print("Downloading QA_RADSAT ", band_file)
        elif str(band) == 'QA_AEROSOL':
            band_file = dir + dir.split('/')[-2] + "_SR_QA_AEROSOL.TIF"
            print("Downloading QA_AEROSOL ", band_file)
        else:
            band_file = dir + dir.split('/')[-2] + "_SR_B" + str(band) + ".TIF"
            print("Downloading Band  ", band_file)
            
        session = boto3.Session(aws_access_key_id= self.aws_access_key, 
                                aws_secret_access_key= self.aws_secret_key, 
                                aws_session_token=self.aws_session_token)
        s3 = session.client('s3')
        files = s3.list_objects_v2(
            Bucket='usgs-landsat',
            Prefix=band_file,
            RequestPayer='requester',
        )['Contents']
        print("Downloading file " + band_file + " to path "  + output_path)
        s3.download_file('usgs-landsat', band_file, output_path, ExtraArgs = {"RequestPayer": "requester"})
        

    def download_tiles_for_band(self, band, year, month, day, lat1, long1, lat2, long2, ignore_qf, ignore_missing = False):
        wrs_res1 = self.get_wrs(lat1, long1)
        path1 = wrs_res1[0]['path']
        row1 = wrs_res1[0]['row']
        wrs_res2 = self.get_wrs(lat2, long2)
        path2 = wrs_res2[0]['path']
        row2 = wrs_res2[0]['row']
        downloaded_files = []
        pixel_files = []
        radsat_files = []
        aerosol_files = []
        for p in range(min(path1, path2), max(path1, path1) + 1 ):
            for r in range(min(row1, row2), max(row1, row2) + 1):
                print("Downloading for band ", band, " Date ", year, "-", month, "-", day, " Path ",  p, " Row ", r)
                dir = self.filter_landsat_data_dir(year, month, day, str(p).zfill(3), str(r).zfill(3))
                if not dir and ignore_missing:
                    print("No data. Ignoring")
                    return []
                print("Data Directory ", dir)
                local_path = "outputs/temp/" + dir.split('/')[-2] + "_SR_B" + str(band) + ".TIF"
                if not os.path.exists(local_path):
                    self.download_landsat_band(dir, band, local_path, ignore_qf)
                downloaded_files.append(local_path)
                
                if ignore_qf:
                    continue
                    
                local_path = "outputs/temp/" + dir.split('/')[-2] + "QA_PIXEL.TIF"
                if not os.path.exists(local_path):
                    self.download_landsat_band(dir, "QA_PIXEL", local_path, ignore_qf)
                pixel_files.append(local_path)
                local_path = "outputs/temp/" + dir.split('/')[-2] + "QA_RADSAT.TIF"
                if not os.path.exists(local_path):
                    self.download_landsat_band(dir, "QA_RADSAT", local_path, ignore_qf)
                radsat_files.append(local_path)
                local_path = "outputs/temp/" + dir.split('/')[-2] + "QA_AEROSOL.TIF"
                if not os.path.exists(local_path):
                    self.download_landsat_band(dir, "QA_AEROSOL", local_path, ignore_qf)
                aerosol_files.append(local_path)
        return downloaded_files, pixel_files, radsat_files, aerosol_files

    def download_tiles_for_band_with_composites(self, band, year, month, day, lat1, long1, lat2, long2, ignore_qf, composite_history_months = 12):
        tiles = []
        tiles.append(self.download_tiles_for_band(band, year, month, day, lat1, long1, lat2, long2, ignore_qf))
        # Download Composites
        for comp in range(composite_history_months):
            if month <= comp:
                t = self.download_tiles_for_band(band, year - 1, 12 - comp + month, 1, lat1, long1, lat2, long2, ignore_qf, ignore_missing = True)
            else:
                t = self.download_tiles_for_band(band, year, month - comp, 1, lat1, long1, lat2, long2, ignore_qf, ignore_missing = True)
            if t:
              tiles.append(t)  
        return tiles

    def process_band_data(self, band, year, month, day, lat1, long1, lat2, long2, output_prefix, ignore_qf):
        b_files = self.download_tiles_for_band_with_composites(band, year, month, day, lat1, long1, lat2, long2, ignore_qf)
        output_files = []
        p_f = []
        r_f = []
        a_f =[]
        for i in range(len(b_files)):
            output_files.append(output_prefix + str(i) + ".TIFF")
            downloaded_files, pixel_files, radsat_files, aerosol_files = b_files[i]
            g = gdal.Warp(output_prefix + str(i) + ".TIFF", 
                          downloaded_files, format="GTiff", options=["COMPRESS=LZW", "TILED=YES"], 
                          resampleAlg='bilinear') # if you want
            g = None
            if ignore_qf:
                continue
                
            p_f.append(output_prefix + str(i) + "P.TIFF")
            g = gdal.Warp(output_prefix + str(i) + "P.TIFF", 
                          pixel_files, format="GTiff", options=["COMPRESS=LZW", "TILED=YES"], 
                          resampleAlg='bilinear', srcNodata= 1, dstNodata = 0) # if you want
            g = None
            r_f.append(output_prefix + str(i) + "R.TIFF")
            g = gdal.Warp(output_prefix + str(i) + "R.TIFF", 
                          radsat_files, format="GTiff", options=["COMPRESS=LZW", "TILED=YES"], 
                          resampleAlg='bilinear', srcNodata= 1, dstNodata = 0) # if you want
            g = None
            a_f.append(output_prefix + str(i) + "A.TIFF")
            g = gdal.Warp(output_prefix + str(i) + "A.TIFF", 
                          aerosol_files, format="GTiff", options=["COMPRESS=LZW", "TILED=YES"], 
                          resampleAlg='bilinear', srcNodata= 1, dstNodata = 0) # if you want
            g = None
        return output_files, p_f, r_f, a_f
    
    def download_all_bands(self, year, month, day, lat1, long1, lat2, long2):
        b3_path = "outputs/B3_Merged_"
        b4_path = "outputs/B4_Merged_"
        b5_path = "outputs/B5_Merged_"

        b3_outputs, b3_p, b3_r, b3_a = self.process_band_data(3, year, month, day, lat1, long1, lat2, long2, b3_path, ignore_qf= False)
        b4_outputs, _, _, _ = self.process_band_data(4, year, month, day, lat1, long1, lat2, long2, b4_path, ignore_qf= True)
        b5_outputs, _, _, _ = self.process_band_data(5, year, month, day, lat1, long1, lat2, long2, b5_path, ignore_qf= True)

        return b3_outputs, b4_outputs, b5_outputs, b3_p, b3_r, b3_a

    def check_values(self, values_to_check, array):
        return (~np.isin(values_to_check, array)).astype(float)

    def mark_cloud2(self, p_px):
        cloud_px_values = [55052, 54852, 54596, 22280]
        #cloud_px_values = [1, 54596, 54852, 55052, 24088, 24216, 24344, 24472, 23888, 23952, 22280, 21826, 21890]
        return self.check_values(p_px, cloud_px_values)

    def mark_cloud(self, p_px):
        #cloud_px_values = [55052, 54852, 54596, 22280]
        #cloud_px_values = [1, 54596, 54852, 55052, 24088, 24216, 24344, 24472, 23888, 23952, 22280, 21826, 21890]
        #cloud_px_values = list(range(1, 21823)) + list(range(21825, 65535))
        cloud_px_values = list(range(1, 21823)) + list(range(21825, 21951)) + list(range(21953, 65535)) #keep 21824 and 21952 (clear land and clear water)
        return self.check_values(p_px, cloud_px_values)

    def mark_aerosol2(self, a_px):
        aerosol_px_values = [192, 194, 196, 224, 228]
        return self.check_values(a_px, aerosol_px_values)

    def mark_aerosol(self, a_px):
        aerosol_px_values = [192, 194, 196, 228]
        return self.check_values(a_px, aerosol_px_values)

    def mark_radsat(self, r_px):
        b3_r_px = self.check_values(r_px & 4, [4])
        b4_r_px = self.check_values(r_px & 8, [8])
        b5_r_px = self.check_values(r_px & 16, [16])
        return b3_r_px, b4_r_px, b5_r_px

    def rescale_bands(self, band_data):
        return band_data * 0.0000275 -0.2
    
class OSMUtils:
    def create_road_raster_flattened(self, lat1, long1, lat2, long2, raster_value = 5,
                           reference_raster = "outputs/ntl.tif",
                           output_path= "outputs/osm.tif", path_thickness = 0.00005):
        with rasterio.open(reference_raster) as f:
                    height = f.height
                    width = f.width
                    crs = f.crs.to_string()
                    transform = f.transform
                    crs = f.crs.to_string()
                    profile = f.profile
        G = ox.graph_from_bbox(max(lat1, lat2), 
                               min(lat1, lat2), 
                               max(long1, long2),
                               min(long1, long2), 
                               retain_all=True,
                               network_type="all")
        ox.save_graph_shapefile(G, "roads")
        roads_shapefile_fn =  "roads/edges.shp"
        road_shapes = []
        with fiona.open(roads_shapefile_fn) as f:
            for row in f:
                geom = row["geometry"]
                #geom = fiona.transform.transform_geom("epsg:4326", crs, geom)
                shape = shapely.geometry.shape(geom)
                shape = shape.buffer(path_thickness) # buffer the linestrings in angles
                road_shapes.append(shapely.geometry.mapping(shape))
        mask = rasterio.features.rasterize(road_shapes, 
                                           out_shape=(height, width), 
                                           fill=0, 
                                           transform=transform, 
                                           all_touched=False, 
                                           default_value=5, 
                                           dtype=np.uint8)
        profile["count"] = 1
        profile["dtype"] = "uint8"
        #profile["nodata"] = 0
        with rasterio.open(output_path, "w", **profile) as f:
            f.write(mask, 1)
    def create_road_raster(self, lat1, long1, lat2, long2, raster_value = 5,
                           reference_raster = "outputs/adjusted_ntl.tif",
                           output_path= "outputs/osm.tif", path_thickness = 0.00005):
        with rasterio.open(reference_raster) as f:
                    height = f.height
                    width = f.width
                    crs = f.crs.to_string()
                    transform = f.transform
                    crs = f.crs.to_string()
                    profile = f.profile
        base = np.zeros((height, width))
        filters = ['["highway"~"primary_link|primary|secondary|secondary_link|tertiary|tertiary_link"]',
                   '["highway"~"motorway|motorway_link"]',
                   '["highway"~"residential"]',
                   '["highway"~"trunk|trunk_link"]',
                   '["highway"~"service|unclassified|road|busway"]'
                  ]
        for idx, filter in enumerate(filters):
            try:
                G = ox.graph_from_bbox(max(lat1, lat2), 
                                       min(lat1, lat2), 
                                       max(long1, long2),
                                       min(long1, long2), 
                                       retain_all=True,
                                       network_type="all", 
                                       custom_filter=filter)
                ox.save_graph_shapefile(G, "roads")
                roads_shapefile_fn =  "roads/edges.shp"
                road_shapes = []
                with fiona.open(roads_shapefile_fn) as f:
                    for row in f:
                        geom = row["geometry"]
                        #geom = fiona.transform.transform_geom("epsg:4326", crs, geom)
                        shape = shapely.geometry.shape(geom)
                        shape = shape.buffer(path_thickness) # buffer the linestrings in angles
                        road_shapes.append(shapely.geometry.mapping(shape))
                mask = rasterio.features.rasterize(road_shapes, 
                                                   out_shape=(height, width), 
                                                   fill=0, 
                                                   transform=transform, 
                                                   all_touched=False, 
                                                   default_value=idx + 1, 
                                                   dtype=np.uint8)
                for row in range(height):
                    for col in range(width):
                        if mask[row][col] > base[row][col]:
                            base[row][col] = mask[row][col]
            except:
                print("No data for filter " + filter)
        profile["count"] = 1
        profile["dtype"] = "uint8"
        #profile["nodata"] = 0
        with rasterio.open(output_path, "w", **profile) as f:
            f.write(base, 1)
            
class VNP46A2Utils:
    def coordinates_to_tile_id(self, lat, longi):
        lat = -lat + 90.0
        longi = longi + 180.0
        tile_v = (lat / 180) * 18
        tile_h = (longi / 360) * 36
        return (int(tile_v), int(tile_h))
    def download_h5(self, earthdata_token, year, month, day, vertical, horizontal, download_file = "VNP46A2.h5"):
        target_coordinate = "h" + str(horizontal).zfill(2) + "v" + str(vertical).zfill(2)
        dt_year = date(year, month, day).timetuple().tm_yday 
        #dt_year = 308
        json_api = "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details/allData/5000/VNP46A2/" + str(year) + "/" + str(dt_year)
        response = requests.get(json_api)    
        dict = response.json()
        download_link  = None
        for elem in dict['content']:
            coordinate = elem['name'].split('.')[2]
            if coordinate == target_coordinate:
                print(elem)
                download_link = elem['downloadsLink']
                break
        if not download_link:
            print("Could not find a VNP46A2 product for coordinate ", target_coordinate)
        else:
            print("Downloading ", download_link)
            headers = {
                'Authorization': f'Bearer {earthdata_token}'
            }
            response = requests.get(download_link, headers=headers)
            if response.status_code == 200:
                with open(download_file, 'wb') as file:
                    file.write(response.content)
                print('File downloaded successfully')
            else:
                print('Failed to download file. Status code:', response.status_code)
                
    def convert_VNP46A2_HDF2TIFF(self, hf5File, outputFile, layer = 0):
        hdflayer = gdal.Open(hf5File, gdal.GA_ReadOnly)
        print(hdflayer.GetSubDatasets()[layer])
        subhdflayer = hdflayer.GetSubDatasets()[layer][0]
        rlayer = gdal.Open(subhdflayer, gdal.GA_ReadOnly)
        rasterFilePre = hf5File[:-3]
        HorizontalTileNumber = int(rlayer.GetMetadata_Dict()["HorizontalTileNumber"])
        VerticalTileNumber = int(rlayer.GetMetadata_Dict()["VerticalTileNumber"])
        WestBoundCoord = (10*HorizontalTileNumber) - 180
        NorthBoundCoord = 90-(10*VerticalTileNumber)
        EastBoundCoord = WestBoundCoord + 10
        SouthBoundCoord = NorthBoundCoord - 10
        EPSG = "-a_srs EPSG:4326" #WGS84
        translateOptionText = EPSG+" -a_ullr " + str(WestBoundCoord) + " " + str(NorthBoundCoord) + " " + str(EastBoundCoord) + " " + str(SouthBoundCoord)
        translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine(translateOptionText))
        gdal.Translate(outputFile,rlayer, options=translateoptions)
