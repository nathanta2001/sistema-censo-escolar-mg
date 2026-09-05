import geopandas as gpd

caminho = r"data/externos/shapefile_regioes_intermediarias/MG_RG_Intermediarias_2025.shp"
gdf = gpd.read_file(caminho)

print("Colunas disponiveis:", list(gdf.columns))
print()
print("Primeiras linhas:")
print(gdf.head(15).drop(columns='geometry'))
print()
print("CRS (sistema de coordenadas):", gdf.crs)
print("Total de regioes:", len(gdf))